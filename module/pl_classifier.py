from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import re
import numpy as np
import os
import pickle
import scipy
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC

from torchmetrics.classification import MulticlassAccuracy, BinaryAUROC, BinaryROC
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC, BinaryROC
from torchmetrics import PearsonCorrCoef  # Accuracy,
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_curve
import monai.transforms as monai_t

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import nibabel as nb

from .models.load_model import load_model
from .utils.metrics import Metrics
from .utils.parser import str2bool
from .utils.losses import NTXentLoss, global_local_temporal_contrastive
from .utils.lr_scheduler import WarmupCosineSchedule, CosineAnnealingWarmUpRestarts

from einops import rearrange

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer


class LitClassifier(pl.LightningModule):
    def __init__(self, data_module, **kwargs):
        super().__init__()
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_hyperparameters({**kwargs, "id": current_time})
        # self.save_hyperparameters(kwargs) # save hyperparameters except data_module (data_module cannot be pickled as a checkpoint)

        # you should define target_values at the Dataset classes
        target_values = data_module.train_dataset.target_values
        if self.hparams.label_scaling_method == 'standardization':
            scaler = StandardScaler()
            normalized_target_values = scaler.fit_transform(target_values)
            print(f'target_mean:{scaler.mean_[0]}, target_std:{scaler.scale_[0]}')
        elif self.hparams.label_scaling_method == 'minmax':
            scaler = MinMaxScaler()
            normalized_target_values = scaler.fit_transform(target_values)
            print(f'target_max:{scaler.data_max_[0]},target_min:{scaler.data_min_[0]}')
        self.scaler = scaler
        print(self.hparams.model)
        self.model = load_model(self.hparams.model, self.hparams)

        if self.hparams.freeze_feature_extractor:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()  # 设置为评估模式
            print("Feature extractor frozen.")

        # Heads
        if not self.hparams.pretraining:
            if self.hparams.downstream_task == 'sex' or self.hparams.downstream_task_type == 'classification' or self.hparams.scalability_check:
                self.output_head = load_model("clf_mlp", self.hparams)
            elif self.hparams.downstream_task == 'age' or self.hparams.downstream_task == 'int_total' or self.hparams.downstream_task == 'int_fluid' or self.hparams.downstream_task_type == 'regression':
                self.output_head = load_model("reg_mlp", self.hparams)
        elif self.hparams.use_contrastive:
            self.output_head = load_model("emb_mlp", self.hparams)
        else:
            raise NotImplementedError("output head should be defined")

        self.metric = Metrics()

        if self.hparams.adjust_thresh:
            self.threshold = 0

    def forward(self, x):
        if self.hparams.freeze_feature_extractor:
            with torch.no_grad():
                features = self.model(x)
        else:
            features = self.model(x)
        return self.output_head(features)

    def augment(self, img):

        B, C, H, W, D, T = img.shape

        device = img.device
        img = rearrange(img, 'b c h w d t -> b t c h w d')

        rand_affine = monai_t.RandAffine(
            prob=1.0,
            # 0.175 rad = 10 degrees
            rotate_range=(0.175, 0.175, 0.175),
            scale_range=(0.1, 0.1, 0.1),
            mode="bilinear",
            padding_mode="border",
            device=device
        )
        rand_noise = monai_t.RandGaussianNoise(prob=0.3, std=0.1)
        rand_smooth = monai_t.RandGaussianSmooth(sigma_x=(0.0, 0.5), sigma_y=(0.0, 0.5), sigma_z=(0.0, 0.5), prob=0.1)
        if self.hparams.augment_only_intensity:
            comp = monai_t.Compose([rand_noise, rand_smooth])
        else:
            comp = monai_t.Compose([rand_affine, rand_noise, rand_smooth])

        for b in range(B):
            aug_seed = torch.randint(0, 10000000, (1,)).item()
            # set augmentation seed to be the same for all time steps
            for t in range(T):
                if self.hparams.augment_only_affine:
                    rand_affine.set_random_state(seed=aug_seed)
                    img[b, t, :, :, :, :] = rand_affine(img[b, t, :, :, :, :])
                else:
                    comp.set_random_state(seed=aug_seed)
                    img[b, t, :, :, :, :] = comp(img[b, t, :, :, :, :])

        img = rearrange(img, 'b t c h w d -> b c h w d t')

        return img

    def _compute_logits(self, batch, augment_during_training=None):
        fmri, subj, target_value, tr, sex = batch.values()

        '''
        # 保存目录
        save_dir = "G:\interpretability_taowu_data"

        def save_fmri_data_grouped(fmri, subj, tr, save_dir):
            """
            将每个病人的 fMRI 数据按帧分组保存到各自的子目录中。

            Args:
                fmri: Tensor of shape [batch_size, 1, 96, 96, 96, num_frames_per_sequence] - 每个病人的 fMRI 数据
                subj: List[str] - 病人 ID 列表
                tr: Tensor/List[int] - 每个病人的起始帧编号
                save_dir: str - 保存数据的根目录
                group_size: int - 每组包含的帧数
                total_frames: int - 每个病人的总帧数
            """

            # 确保保存目录存在
            os.makedirs(save_dir, exist_ok=True)

            for idx, subj_id in enumerate(subj):
                fmri_sequence = fmri[idx]  # 单个病人的 fMRI 序列 [1, 96, 96, 96, num_frames_per_sequence]
                start_frame = tr[idx]  # 当前病人的起始帧编号

                # 动态创建病人的子目录
                subj_dir = os.path.join(save_dir, subj_id)
                os.makedirs(subj_dir, exist_ok=True)
                save_path = os.path.join(subj_dir, f"{start_frame}_{start_frame+20-1}")
                torch.save(fmri_sequence, os.path.join(subj_dir, f"{save_path}.pt"))
                print(f"Saved {subj_id}'s frames {start_frame}_{start_frame+20-1} to {save_path}.pt")



        save_fmri_data_grouped(fmri, subj, tr.cpu().tolist(), save_dir)
        '''



        if augment_during_training:
            fmri = self.augment(fmri)
        if not self.training:  # 验证或测试阶段
            with torch.no_grad():
                feature = self.model(fmri)
        elif self.hparams.freeze_feature_extractor:
            for param in self.model.parameters():
                param.requires_grad = False
            with torch.no_grad():
                feature = self.model(fmri)
        else:
            feature = self.model(fmri)
        # Classification task
        if self.hparams.downstream_task == 'sex' or self.hparams.downstream_task_type == 'classification' or self.hparams.scalability_check:
            logits = self.output_head(feature).squeeze()  # self.clf(feature).squeeze()
            # target = target_value.float().squeeze()
            target = target_value.long().squeeze()  # 确保 target 是 long 类型且是一维

        # Regression task
        elif self.hparams.downstream_task == 'age' or self.hparams.downstream_task == 'int_total' or self.hparams.downstream_task == 'int_fluid' or self.hparams.downstream_task_type == 'regression':
            # target_mean, target_std = self.determine_target_mean_std()
            logits = self.output_head(feature)  # (batch,1) or # tuple((batch,1), (batch,1))
            unnormalized_target = target_value.float()  # (batch,1)
            if self.hparams.label_scaling_method == 'standardization':  # default
                target = (unnormalized_target - self.scaler.mean_[0]) / (self.scaler.scale_[0])
            elif self.hparams.label_scaling_method == 'minmax':
                target = (unnormalized_target - self.scaler.data_min_[0]) / (
                            self.scaler.data_max_[0] - self.scaler.data_min_[0])

        return subj, logits, target

    def _calculate_loss(self, batch, mode):
        if self.hparams.pretraining:
            fmri, subj, target_value, tr, sex = batch.values()

            cond1 = (self.hparams.in_chans == 1 and not self.hparams.with_voxel_norm)
            assert cond1, "Wrong combination of options"
            loss = 0

            if self.hparams.use_contrastive:
                assert self.hparams.contrastive_type != "none", "Contrastive type not specified"

                # B, C, H, W, D, T = image shape
                y, diff_y = fmri

                batch_size = y.shape[0]
                if (len(subj) != len(tuple(subj))) and mode == 'train':
                    print('Some sub-sequences in a batch came from the same subject!')
                criterion = NTXentLoss(device='cuda', batch_size=batch_size,
                                       temperature=self.hparams.temperature,
                                       use_cosine_similarity=True).cuda()
                criterion_ll = NTXentLoss(device='cuda', batch_size=2,
                                          temperature=self.hparams.temperature,
                                          use_cosine_similarity=True).cuda()

                # type 1: IC
                # type 2: LL
                # type 3: IC + LL
                if self.hparams.contrastive_type in [1, 3]:
                    out_global_1 = self.output_head(self.model(self.augment(y)), "g")
                    out_global_2 = self.output_head(self.model(self.augment(diff_y)), "g")
                    ic_loss = criterion(out_global_1, out_global_2)
                    loss += ic_loss

                if self.hparams.contrastive_type in [2, 3]:
                    out_local_1 = []
                    out_local_2 = []
                    out_local_swin1 = self.model(self.augment(y))
                    out_local_swin2 = self.model(self.augment(y))
                    out_local_1.append(self.output_head(out_local_swin1, "l"))
                    out_local_2.append(self.output_head(out_local_swin2, "l"))

                    out_local_swin1 = self.model(self.augment(diff_y))
                    out_local_swin2 = self.model(self.augment(diff_y))
                    out_local_1.append(self.output_head(out_local_swin1, "l"))
                    out_local_2.append(self.output_head(out_local_swin2, "l"))

                    ll_loss = 0
                    # loop over batch size
                    for i in range(out_local_1[0].shape[0]):
                        # out_local shape should be: BS, n_local_clips, D
                        ll_loss += criterion_ll(torch.stack(out_local_1, dim=1)[i],
                                                torch.stack(out_local_2, dim=1)[i])
                    loss += ll_loss

                result_dict = {
                    f"{mode}_loss": loss,
                }
        else:
            subj, logits, target = self._compute_logits(batch,
                                                        augment_during_training=self.hparams.augment_during_training)

            if self.hparams.downstream_task == 'sex' or self.hparams.downstream_task_type == 'classification' or self.hparams.scalability_check:
                if self.hparams.dataset_name == 'olfactory':
                    # loss = F.binary_cross_entropy_with_logits(logits, target) # target is float
                    loss = F.cross_entropy(logits, target.long())  # 三分类任务
                    # acc = self.metric.get_accuracy_binary(logits, target.float().squeeze())
                    acc = self.metric.get_accuracy(logits, target)
                    result_dict = {
                        f"{mode}_loss": loss,
                        f"{mode}_acc": acc,
                    }
                elif self.hparams.dataset_name == 'taowu':

                    '''
                    # 病人聚合
                    # 定义一个字典存储每个病人的 logits 和 targets
                    subj_logits_dict = {}
                    subj_targets_dict = {}

                    # 遍历每个病人的数据
                    for i, subj_id in enumerate(subj):
                        if subj_id not in subj_logits_dict:
                            subj_logits_dict[subj_id] = []
                            subj_targets_dict[subj_id] = target[i]  # 直接存储原始张量，保留计算图
                        subj_logits_dict[subj_id].append(logits[i])  # 存储 logits，保留计算图

                    # 计算每个病人的平均 logits 和 targets
                    subj_avg_logits = []
                    subj_targets = []

                    for subj_id in subj_logits_dict:
                        avg_logit = torch.stack(subj_logits_dict[subj_id]).mean()  # 使用 torch.stack 保留计算图
                        subj_avg_logits.append(avg_logit)
                        subj_targets.append(subj_targets_dict[subj_id])  # 保留原始 targets

                    # 将结果转换为张量
                    subj_avg_logits = torch.stack(subj_avg_logits).to(logits.device)  # 保留计算图
                    subj_targets = torch.stack(subj_targets).to(logits.device).float()  # 转换 targets 为 float

                    # 计算损失
                    loss = F.binary_cross_entropy_with_logits(subj_avg_logits, subj_targets)

                    # 计算准确率
                    probs = torch.sigmoid(subj_avg_logits)
                    acc = ((probs > 0.5).long() == subj_targets.long()).float().mean()

                    # 将结果存储到日志字典
                    result_dict = {
                        f"{mode}_loss": loss,
                        f"{mode}_acc": acc,
                    }

                    # 记录日志
                self.log_dict(result_dict, prog_bar=True, sync_dist=False, add_dataloader_idx=False, on_step=True,
                                  on_epoch=True,
                                  batch_size=self.hparams.batch_size)

                '''
                    loss = F.binary_cross_entropy_with_logits(logits, target.float())  # 使用 BCE 损失函数
                    acc = self.metric.get_accuracy_binary(logits, target)  # 二分类准确率
                    result_dict = {
                        f"{mode}_loss": loss,
                        f"{mode}_acc": acc,
                    }



                elif self.hparams.dataset_name == 'ABIDE':
                    loss = F.binary_cross_entropy_with_logits(logits, target.float())  # 使用 BCE 损失函数
                    acc = self.metric.get_accuracy_binary(logits, target)  # 二分类准确率
                    result_dict = {
                        f"{mode}_loss": loss,
                        f"{mode}_acc": acc,
                    }

                self.log_dict(result_dict, prog_bar=True, sync_dist=False, add_dataloader_idx=False, on_step=True,
                              on_epoch=True, batch_size=self.hparams.batch_size)



            elif self.hparams.downstream_task == 'age' or self.hparams.downstream_task == 'int_total' or self.hparams.downstream_task == 'int_fluid' or self.hparams.downstream_task_type == 'regression':
                loss = F.mse_loss(logits.squeeze(), target.squeeze())
                l1 = F.l1_loss(logits.squeeze(), target.squeeze())
                result_dict = {
                    f"{mode}_loss": loss,
                    f"{mode}_mse": loss,
                    f"{mode}_l1_loss": l1
                }
        self.log_dict(result_dict, prog_bar=True, sync_dist=False, add_dataloader_idx=False, on_step=True,
                      on_epoch=True, batch_size=self.hparams.batch_size)  # batch_size = batch_size
        return loss

    def _evaluate_metrics(self, subj_array, total_out, mode):
        if self.hparams.dataset_name == 'olfactory':

            # 从 total_out 中提取 logits 和目标标签
            logits = total_out[:, :-1]  # 提取 logits，形状为 (N, num_classes)
            targets = total_out[:, -1].long()  # 提取目标标签，并转换为 long 类型

            # 使用 softmax 转换 logits 为概率
            probs = logits.softmax(dim=1)  # 转换为概率分布

            # 分类任务的准确率计算（用 softmax 后的概率进行 argmax）
            acc = (probs.argmax(dim=1) == targets).float().mean()

            # 计算 AUROC
            auroc_func = MulticlassAUROC(num_classes=logits.shape[1], average='macro').to(logits.device)
            auroc = auroc_func(probs, targets)  # 用 softmax 转换后的概率计算 AUROC

            print(f"{mode}_acc", acc)
            print(f"{mode}_auroc", auroc)

            # 如果需要日志输出
            self.log(f"{mode}_acc", acc, sync_dist=True)
            self.log(f"{mode}_AUROC", auroc, sync_dist=True)

            # 如果需要计算损失（例如用于评估）
            loss = F.cross_entropy(logits, targets)  # CrossEntropyLoss 直接接受 logits，无需 softmax

            # 日志记录损失
            print(f"{mode}_loss", loss)
            self.log(f"{mode}_loss", loss, sync_dist=True)

        elif self.hparams.dataset_name == 'ABIDE':
            # 提取 logits 和 targets
            logits = total_out[:, 0]  # 二分类只需要一列 logits
            targets = total_out[:, 1].long()

            # 创建一个字典来存储每个病人的 logits 和 targets
            subj_logits_dict = {}
            subj_targets_dict = {}

            for i, subj in enumerate(subj_array):
                if subj not in subj_logits_dict:
                    subj_logits_dict[subj] = []
                    subj_targets_dict[subj] = targets[i].item()  # 每个病人的目标值是固定的
                subj_logits_dict[subj].append(logits[i].item())

            # 计算每个病人的平均 logits
            subj_avg_logits = []
            subj_targets = []

            for subj in subj_logits_dict:
                avg_logit = torch.tensor(subj_logits_dict[subj]).mean().item()
                subj_avg_logits.append(avg_logit)
                subj_targets.append(subj_targets_dict[subj])

            # 转换为 tensor
            subj_avg_logits = torch.tensor(subj_avg_logits, device=logits.device)
            subj_targets = torch.tensor(subj_targets, device=logits.device).long()

            # 如果需要调整阈值
            if self.hparams.adjust_thresh:
                # 找到最佳平衡准确率
                best_bal_acc = 0
                best_thresh = 0
                for thresh in np.arange(-5, 5, 0.01):
                    bal_acc = balanced_accuracy_score(subj_targets.cpu(), (subj_avg_logits >= thresh).int().cpu())
                    if bal_acc > best_bal_acc:
                        best_bal_acc = bal_acc
                        best_thresh = thresh
                self.log(f"{mode}_best_thresh", best_thresh, sync_dist=True)
                self.log(f"{mode}_best_balacc", best_bal_acc, sync_dist=True)
                print('\n')
                print(f"{mode}_best_balacc", best_bal_acc)
                print(f"{mode}_best_balacc", best_bal_acc)

                # 计算 Youden's J statistic 阈值
                fpr, tpr, thresholds = roc_curve(subj_targets.cpu(), subj_avg_logits.cpu())
                youden_idx = np.argmax(tpr - fpr)
                youden_thresh = thresholds[youden_idx]
                self.log(f"{mode}_youden_thresh", youden_thresh, sync_dist=True)
                self.log(f"{mode}_youden_balacc",
                         balanced_accuracy_score(subj_targets.cpu(), (subj_avg_logits >= youden_thresh).int().cpu()),
                         sync_dist=True)
                print(f"{mode}_youden_thresh", youden_thresh)
                print(f"{mode}_youden_balacc", balanced_accuracy_score(subj_targets.cpu(), (subj_avg_logits >= youden_thresh).int().cpu()))

                # 保存阈值并在测试时使用验证集的最佳阈值
                if mode == 'valid':
                    self.threshold = youden_thresh
                elif mode == 'test':
                    bal_acc = balanced_accuracy_score(subj_targets.cpu(),
                                                      (subj_avg_logits >= self.threshold).int().cpu())
                    self.log(f"{mode}_balacc_from_valid_thresh", bal_acc, sync_dist=True)
                    print(f"{mode}_balacc_from_valid_thresh", bal_acc)

            # 计算准确率
            probs = torch.sigmoid(subj_avg_logits)  # 使用 sigmoid 转换平均 logits
            acc = ((probs > 0.5).long() == subj_targets).float().mean()

            # 计算 AUROC
            auroc_func = BinaryAUROC().to(subj_avg_logits.device)
            auroc = auroc_func(probs, subj_targets)

            # 日志记录
            self.log(f"{mode}_acc", acc, sync_dist=True)
            print(f"{mode}_acc", acc)
            self.log(f"{mode}_AUROC", auroc, sync_dist=True)
            print(f"{mode}_AUROC", auroc)

            # 计算损失
            loss = F.binary_cross_entropy_with_logits(subj_avg_logits, subj_targets.float())
            self.log(f"{mode}_loss", loss, sync_dist=True)
            print(f"{mode}_loss", loss)

            return acc, auroc, loss
    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if self.hparams.pretraining:
            if dataloader_idx == 0:
                self._calculate_loss(batch, mode="valid")
            else:
                self._calculate_loss(batch, mode="test")
        else:
            subj, logits, target = self._compute_logits(batch)
            if self.hparams.downstream_task_type == 'multi_task':
                output = torch.stack([logits[1].squeeze(), target], dim=1)  # logits[1] : regression head
            else:
                # 直接返回 logits 和 target 以避免 stack 的形状不匹配问题
                return (subj, logits.detach().cpu(), target.float().detach().cpu())

    def validation_epoch_end(self, outputs):
        if not self.hparams.pretraining:
            if self.hparams.dataset_name == 'olfactory':

                # 解析验证集和测试集的输出结果
                outputs_valid = outputs[0]
                outputs_test = outputs[1]

                # 初始化存储验证集和测试集的数据
                subj_valid = []
                subj_test = []
                out_valid_list = []
                out_test_list = []

                # 提取验证集的 subject 列表和 logits、target
                for output in outputs_valid:
                    subj, logits, target = output  # validation_step 应返回三个元素 (subj, logits, target)
                    subj_valid += subj
                    # 将 logits 和 target 拼接到一起，确保输出格式一致
                    out_valid_list.append(torch.cat([logits, target.unsqueeze(1)], dim=1))

                # 提取测试集的 subject 列表和 logits、target
                for output in outputs_test:
                    subj, logits, target = output
                    subj_test += subj
                    out_test_list.append(torch.cat([logits, target.unsqueeze(1)], dim=1))

                # 将 subject 列表转换为 numpy 数组，将预测输出结果合并为张量
                subj_valid = np.array(subj_valid)
                subj_test = np.array(subj_test)
                total_out_valid = torch.cat(out_valid_list, dim=0)
                total_out_test = torch.cat(out_test_list, dim=0)

                # 将验证集的预测值和真实标签保存为文件
                valid_predictions = pd.DataFrame({
                    'subject': subj_valid,
                    'predicted': total_out_valid[:, :-1].softmax(dim=1).argmax(dim=1).cpu().numpy(),  # 转换为预测类别
                    'true_label': total_out_valid[:, -1].cpu().numpy().astype(int)  # 确保标签是整型
                })

                test_predictions = pd.DataFrame({
                    'subject': subj_test,
                    'predicted': total_out_test[:, :-1].softmax(dim=1).argmax(dim=1).cpu().numpy(),
                    'true_label': total_out_test[:, -1].cpu().numpy().astype(int)
                })

                # 文件保存路径
                save_dir = os.path.join('predictions', str(self.hparams.id))
                os.makedirs(save_dir, exist_ok=True)

                # 保存验证集结果到 CSV 文件
                valid_file_path = os.path.join(save_dir, f'valid_predictions_epoch_{self.current_epoch}.csv')
                valid_predictions.to_csv(valid_file_path, index=False)
                print(f"Validation predictions saved to {valid_file_path}")

                # 保存测试集结果到 CSV 文件
                test_file_path = os.path.join(save_dir, f'test_predictions_epoch_{self.current_epoch}.csv')
                test_predictions.to_csv(test_file_path, index=False)
                print(f"Test predictions saved to {test_file_path}")

                # 调用评估函数计算验证集和测试集的评估指标
                self._evaluate_metrics(subj_valid, total_out_valid, mode="valid")
                self._evaluate_metrics(subj_test, total_out_test, mode="test")
            elif self.hparams.dataset_name == 'ABIDE':

                outputs_valid = outputs[0]
                outputs_test = outputs[1]

                # 初始化存储验证集和测试集的数据
                subj_valid = []
                subj_test = []
                out_valid_list = []
                out_test_list = []

                # 提取验证集的 subject 列表和 logits、target
                for output in outputs_valid:
                    subjects, logits, target = output  # 从三元组中解包
                    subj_valid += subjects  # 扩展 subject 列表
                    out_valid_list.append(
                        torch.cat([logits.unsqueeze(1), target.unsqueeze(1)], dim=1))  # 拼接 logits 和 target

                # 提取测试集的 subject 列表和 logits、target
                for output in outputs_test:
                    subjects, logits, target = output
                    subj_test += subjects
                    out_test_list.append(torch.cat([logits.unsqueeze(1), target.unsqueeze(1)], dim=1))


                # 将 subject 列表转换为 numpy 数组，将预测输出结果合并为张量
                subj_valid = np.array(subj_valid)
                subj_test = np.array(subj_test)
                total_out_valid = torch.cat(out_valid_list, dim=0)
                total_out_test = torch.cat(out_test_list, dim=0)
                '''
                # 将验证集的预测值和真实标签保存为文件
                valid_predictions = pd.DataFrame({
                    'subject': subj_valid,
                    'predicted': total_out_valid[:, :-1].softmax(dim=1).argmax(dim=1).cpu().numpy(),  # 转换为预测类别
                    'true_label': total_out_valid[:, -1].cpu().numpy().astype(int)  # 确保标签是整型
                })

                test_predictions = pd.DataFrame({
                    'subject': subj_test,
                    'predicted': total_out_test[:, :-1].softmax(dim=1).argmax(dim=1).cpu().numpy(),
                    'true_label': total_out_test[:, -1].cpu().numpy().astype(int)
                })

                # 文件保存路径
                save_dir = os.path.join('predictions', str(self.hparams.id))
                os.makedirs(save_dir, exist_ok=True)

                # 保存验证集结果到 CSV 文件
                valid_file_path = os.path.join(save_dir, f'valid_predictions_epoch_{self.current_epoch}.csv')
                valid_predictions.to_csv(valid_file_path, index=False)
                print(f"Validation predictions saved to {valid_file_path}")

                # 保存测试集结果到 CSV 文件
                test_file_path = os.path.join(save_dir, f'test_predictions_epoch_{self.current_epoch}.csv')
                test_predictions.to_csv(test_file_path, index=False)
                print(f"Test predictions saved to {test_file_path}")
                '''

                # 调用评估函数计算验证集和测试集的评估指标
                self._evaluate_metrics(subj_valid, total_out_valid, mode="valid")
                self._evaluate_metrics(subj_test, total_out_test, mode="test")


    # If you use loggers other than Neptune you may need to modify this
    def _save_predictions(self, total_subjs, total_out, mode):
        self.subject_accuracy = {}
        for subj, output in zip(total_subjs, total_out):
            if self.hparams.downstream_task == 'sex':
                score = torch.sigmoid(output[0]).item()
            else:
                score = output[0].item()

            if subj not in self.subject_accuracy:
                self.subject_accuracy[subj] = {'score': [score], 'mode': mode, 'truth': output[1], 'count': 1}
            else:
                self.subject_accuracy[subj]['score'].append(score)
                self.subject_accuracy[subj]['count'] += 1

        if self.hparams.strategy == None:
            pass
        elif 'ddp' in self.hparams.strategy and len(self.subject_accuracy) > 0:
            world_size = torch.distributed.get_world_size()
            total_subj_accuracy = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(total_subj_accuracy,
                                                self.subject_accuracy)  # gather and broadcast to whole ranks
            accuracy_dict = {}
            for dct in total_subj_accuracy:
                for subj, metric_dict in dct.items():
                    if subj not in accuracy_dict:
                        accuracy_dict[subj] = metric_dict
                    else:
                        accuracy_dict[subj]['score'] += metric_dict['score']
                        accuracy_dict[subj]['count'] += metric_dict['count']
            self.subject_accuracy = accuracy_dict
        if self.trainer.is_global_zero:
            for subj_name, subj_dict in self.subject_accuracy.items():
                subj_pred = np.mean(subj_dict['score'])
                subj_error = np.std(subj_dict['score'])
                subj_truth = subj_dict['truth'].item()
                subj_count = subj_dict['count']
                subj_mode = subj_dict['mode']  # train, val, test

                # only save samples at rank 0 (total iterations/world_size numbers are saved)
                os.makedirs(os.path.join('predictions', self.hparams.id), exist_ok=True)
                with open(os.path.join('predictions', self.hparams.id, 'iter_{}.txt'.format(self.current_epoch)),
                          'a+') as f:
                    f.write('subject:{} ({})\ncount: {} outputs: {:.4f}\u00B1{:.4f}  -  truth: {}\n'.format(subj_name,
                                                                                                            subj_mode,
                                                                                                            subj_count,
                                                                                                            subj_pred,
                                                                                                            subj_error,
                                                                                                            subj_truth))

            with open(os.path.join('predictions', self.hparams.id, 'iter_{}.pkl'.format(self.current_epoch)),
                      'wb') as fw:
                pickle.dump(self.subject_accuracy, fw)

    def test_step(self, batch, batch_idx):
        subj, logits, target = self._compute_logits(batch)
        # 不拼接 logits 和 target，直接返回它们，以便后续处理
        return (subj, logits.detach().cpu(), target.detach().cpu())

    def test_epoch_end(self, outputs):
        if not self.hparams.pretraining:
            if self.hparams.dataset_name == 'olfactory':

                subj_test = []
                out_test_list = []

                # 遍历每个 batch 的输出
                for output in outputs:
                    # 检查 output 的长度，确保它包含我们需要的两部分
                    if len(output) == 2:
                        subj, out = output
                    elif len(output) == 3:
                        subj, logits, target = output
                        out = torch.cat([logits, target.unsqueeze(1)], dim=1)
                    else:
                        raise ValueError("Unexpected output format in test_step output.")

                    subj_test += subj
                    out_test_list.append(out)

                # 将 subject 列表转换为 numpy 数组，将预测输出结果合并为张量
                subj_test = np.array(subj_test)
                total_out_test = torch.cat(out_test_list, dim=0)

                # 调用评估函数计算测试集的评估指标
                self._evaluate_metrics(subj_test, total_out_test, mode="test")

            elif self.hparams.dataset_name == 'ABIDE':
                subj_test = []
                out_test_list = []

                # 遍历每个 batch 的输出
                for output in outputs:
                    # 解包三元组
                    subjects, logits, target = output

                    # 打印调试信息，检查 logits 和 target 的形状
                    print(f"logits shape: {logits.shape}, target shape: {target.shape}")

                    # 确保 logits 和 target 是一维的
                    logits = logits.view(-1)  # logits 应该是 [B]
                    target = target.view(-1)  # target 应该是 [B]

                    # 收集 subjects
                    subj_test += subjects

                    # 拼接 logits 和 target，形成 (B, 2) 的张量
                    out = torch.cat([logits.unsqueeze(1), target.unsqueeze(1)], dim=1)
                    out_test_list.append(out)

                # 将 subject 列表转换为 numpy 数组
                subj_test = np.array(subj_test)

                # 将预测输出结果合并为一个张量
                total_out_test = torch.cat(out_test_list, dim=0)

                # 调用评估函数计算测试集的评估指标
                self._evaluate_metrics(subj_test, total_out_test, mode="test")


    def on_train_epoch_start(self) -> None:
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.total_time = 0
        self.repetitions = 200
        self.gpu_warmup = 50
        self.timings = np.zeros((self.repetitions, 1))
        return super().on_train_epoch_start()

    def on_train_batch_start(self, batch, batch_idx):
        if self.hparams.scalability_check:
            if batch_idx < self.gpu_warmup:
                pass
            elif (batch_idx - self.gpu_warmup) < self.repetitions:
                self.starter.record()
        return super().on_train_batch_start(batch, batch_idx)

    def on_train_batch_end(self, out, batch, batch_idx):
        if self.hparams.scalability_check:
            if batch_idx < self.gpu_warmup:
                pass
            elif (batch_idx - self.gpu_warmup) < self.repetitions:
                self.ender.record()
                torch.cuda.synchronize()
                curr_time = self.starter.elapsed_time(self.ender) / 1000
                self.total_time += curr_time
                self.timings[batch_idx - self.gpu_warmup] = curr_time
            elif (batch_idx - self.gpu_warmup) == self.repetitions:
                mean_syn = np.mean(self.timings)
                std_syn = np.std(self.timings)

                Throughput = (self.repetitions * self.hparams.batch_size * int(self.hparams.num_nodes) * int(
                    self.hparams.devices)) / self.total_time

                self.log(f"Throughput", Throughput, sync_dist=False)
                self.log(f"mean_time", mean_syn, sync_dist=False)
                self.log(f"std_time", std_syn, sync_dist=False)
                print('mean_syn:', mean_syn)
                print('std_syn:', std_syn)

        return super().on_train_batch_end(out, batch, batch_idx)

    # def on_before_optimizer_step(self, optimizer, optimizer_idx: int) -> None:

    def configure_optimizers(self):
        if self.hparams.optimizer == "AdamW":
            optim = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == "SGD":
            optim = torch.optim.SGD(
                self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay,
                momentum=self.hparams.momentum
            )
        else:
            print("Error: Input a correct optimizer name (default: AdamW)")

        if self.hparams.use_scheduler:
            print()
            print("training steps: " + str(self.trainer.estimated_stepping_batches))
            print("using scheduler")
            print()
            total_iterations = self.trainer.estimated_stepping_batches  # ((number of samples/batch size)/number of gpus) * num_epochs
            gamma = self.hparams.gamma
            base_lr = self.hparams.learning_rate
            warmup = int(total_iterations * 0.05)  # adjust the length of warmup here.
            T_0 = int(self.hparams.cycle * total_iterations)
            T_mult = 1

            sche = CosineAnnealingWarmUpRestarts(optim, first_cycle_steps=T_0, cycle_mult=T_mult, max_lr=base_lr,
                                                 min_lr=1e-9, warmup_steps=warmup, gamma=gamma)
            print('total iterations:', self.trainer.estimated_stepping_batches * self.hparams.max_epochs)

            scheduler = {
                "scheduler": sche,
                "name": "lr_history",
                "interval": "step",
            }

            return [optim], [scheduler]
        else:
            return optim

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("Default classifier")

        # training related
        group.add_argument("--grad_clip", action='store_true', help="whether to use gradient clipping")
        group.add_argument("--optimizer", type=str, default="AdamW", help="which optimizer to use [AdamW, SGD]")
        group.add_argument("--use_scheduler", default=True, help="whether to use scheduler")
        group.add_argument("--weight_decay", type=float, default=0.01, help="weight decay for optimizer")
        group.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate for optimizer")
        group.add_argument("--momentum", type=float, default=0, help="momentum for SGD")
        group.add_argument("--gamma", type=float, default=1.0, help="decay for exponential LR scheduler")
        group.add_argument("--cycle", type=float, default=0.3, help="cycle size for CosineAnnealingWarmUpRestarts")
        group.add_argument("--milestones", nargs="+", default=[100, 150], type=int, help="lr scheduler")
        group.add_argument("--adjust_thresh", default=True, help="whether to adjust threshold for valid/test")

        # pretraining-related
        # group.add_argument("--use_contrastive", action='store_true', help="whether to use contrastive learning (specify --contrastive_type argument as well)")
        group.add_argument("--use_contrastive", default=False,
                           help="whether to use contrastive learning (specify --contrastive_type argument as well)")
        group.add_argument("--contrastive_type", default=1, type=int,
                           help="combination of contrastive losses to use [1: Use the Instance contrastive loss function, 2: Use the local-local temporal contrastive loss function, 3: Use the sum of both loss functions]")
        group.add_argument("--pretraining", default=False, help="whether to use pretraining")
        group.add_argument("--augment_during_training", default=False,
                           help="whether to augment input images during training")
        group.add_argument("--augment_only_affine", action='store_true',
                           help="whether to only apply affine augmentation")
        group.add_argument("--augment_only_intensity", action='store_true',
                           help="whether to only apply intensity augmentation")
        group.add_argument("--temperature", default=0.1, type=float, help="temperature for NTXentLoss")

        # model related
        group.add_argument("--model", type=str, default="swin4d_ver7", help="which model to be used")
        group.add_argument("--in_chans", type=int, default=1, help="Channel size of input image")
        group.add_argument("--embed_dim", type=int, default=24, help="embedding size (recommend to use 24, 36, 48)")
        group.add_argument("--window_size", nargs="+", default=[4, 4, 4, 4], type=int,
                           help="window size from the second layers")
        group.add_argument("--first_window_size", nargs="+", default=[2, 2, 2, 2], type=int, help="first window size")
        group.add_argument("--patch_size", nargs="+", default=[4, 4, 4, 1], type=int, help="patch size")
        group.add_argument("--depths", nargs="+", default=[2, 2, 6, 2], type=int, help="depth of layers in each stage")
        group.add_argument("--num_heads", nargs="+", default=[3, 6, 12, 24], type=int, help="The number of heads for each attention layer")
        group.add_argument("--c_multiplier", type=int, default=2,
                           help="channel multiplier for Swin Transformer architecture")
        group.add_argument("--last_layer_full_MSA", type=str2bool, default=True,
                           help="whether to use full-scale multi-head self-attention at the last layers")
        group.add_argument("--attn_drop_rate", type=float, default=0.1, help="dropout rate of attention layers")
        group.add_argument("--clf_head_version", type=str, default="multi", help="clf head version, v2 has a hidden layer")
        group.add_argument("--roi_template_path", type=str, default="D:\\Project\\Parkinson\\SwiFT-main\\project\\roi_template\\AAL_61x73x61_YCG.nii", help="roi template path")
        # others
        group.add_argument("--scalability_check", action='store_true', help="whether to check scalability")
        group.add_argument("--process_code", default=None,
                           help="Slurm code/PBS code. Use this argument if you want to save process codes to your log")

        return parser
