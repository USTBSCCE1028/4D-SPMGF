import os
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from .data_preprocess_and_load.datasets import S1200, ABCD, UKB, Dummy, OlfactoryDataset, taowuDataset, ABIDEDataset
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from .parser import str2bool

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

class fMRIDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        # generate splits folder
        if self.hparams.pretraining:
                split_dir_path = f'./data/splits/{self.hparams.dataset_name}/pretraining'
        else:
            split_dir_path = f'./data/splits/{self.hparams.dataset_name}'
        os.makedirs(split_dir_path, exist_ok=True)
        self.split_file_path = os.path.join(split_dir_path, f"split_fixed_{self.hparams.dataset_split_num}.txt")
        
        self.setup()

        #pl.seed_everything(seed=self.hparams.data_seed)

    def get_dataset(self):
        if self.hparams.dataset_name == "Dummy":
            return Dummy
        elif self.hparams.dataset_name == "S1200":
            return S1200
        elif self.hparams.dataset_name == "ABCD":
            return ABCD
        elif self.hparams.dataset_name == 'UKB':
            return UKB
        elif self.hparams.dataset_name == "olfactory":
            return OlfactoryDataset
        elif self.hparams.dataset_name == "taowu":
            return taowuDataset
        elif self.hparams.dataset_name == "ABIDE":
            return ABIDEDataset
        else:
            raise NotImplementedError

    def convert_subject_list_to_idx_list(self, train_names, val_names, test_names, subj_list):
        if self.hparams.dataset_name == "olfactory":
            #subj_idx = np.array([str(x[0]) for x in subj_list])
            subj_idx = np.array([str(x[1]) for x in subj_list])
            S = np.unique([x[1] for x in subj_list])
            # print(S)
            print('unique subjects:',len(S))
            train_idx = np.where(np.in1d(subj_idx, train_names))[0].tolist()
            val_idx = np.where(np.in1d(subj_idx, val_names))[0].tolist()
            test_idx = np.where(np.in1d(subj_idx, test_names))[0].tolist()

            return train_idx, val_idx, test_idx
        elif self.hparams.dataset_name == 'taowu':

            subj_idx = np.array([str(x[1]) for x in subj_list])  # 提取 subject 名称
            unique_subjects = np.unique([x[1] for x in subj_list])
            print('unique subjects:', len(unique_subjects))

            train_idx = np.where(np.in1d(subj_idx, train_names))[0].tolist()
            val_idx = np.where(np.in1d(subj_idx, val_names))[0].tolist()
            test_idx = np.where(np.in1d(subj_idx, test_names))[0].tolist()

            return train_idx, val_idx, test_idx

        elif self.hparams.dataset_name == 'ABIDE':

            subj_idx = np.array([str(x[1]) for x in subj_list])  # 提取 subject 名称
            unique_subjects = np.unique([x[1] for x in subj_list])
            print('unique subjects:', len(unique_subjects))

            train_idx = np.where(np.in1d(subj_idx, train_names))[0].tolist()
            val_idx = np.where(np.in1d(subj_idx, val_names))[0].tolist()
            test_idx = np.where(np.in1d(subj_idx, test_names))[0].tolist()

            return train_idx, val_idx, test_idx

    def save_split(self, sets_dict):
        if self.hparams.dataset_name == "olfactory":
            with open(self.split_file_path, "w+") as f:
                for name, subj_list in sets_dict.items():
                    f.write(name + "\n")
                    for subj_name in subj_list:
                        f.write(str(subj_name) + "\n")

        elif self.hparams.dataset_name == 'taowu':
            with open(self.split_file_path, "w+") as f:
                for name, subj_list in sets_dict.items():
                    f.write(name + "\n")
                    for subj_name in subj_list:
                        f.write(str(subj_name) + "\n")

        elif self.hparams.dataset_name == 'ABIDE':
            with open(self.split_file_path, "w+") as f:
                for name, subj_list in sets_dict.items():
                    f.write(name + "\n")
                    for subj_name in subj_list:
                        f.write(str(subj_name) + "\n")


    def determine_split_randomly(self, S):

        if self.hparams.dataset_name == "olfactory":
            # 将每个类别的subject分组
            ctl_subjects = [subj for subj in S.keys() if "CTL" in subj]
            odn_subjects = [subj for subj in S.keys() if "ODN" in subj]
            odp_subjects = [subj for subj in S.keys() if "ODP" in subj]

            # 根据类别分别划分训练、验证和测试集
            def split_group_olfactory(subjects):
                n_train = int(len(subjects) * self.hparams.train_split)
                n_val = int(len(subjects) * self.hparams.val_split)
                train = np.random.choice(subjects, n_train, replace=False)
                remaining = np.setdiff1d(subjects, train)
                val = np.random.choice(remaining, n_val, replace=False)
                test = np.setdiff1d(subjects, np.concatenate([train, val]))
                return train, val, test

            ctl_train, ctl_val, ctl_test = split_group_olfactory(ctl_subjects)
            odn_train, odn_val, odn_test = split_group_olfactory(odn_subjects)
            odp_train, odp_val, odp_test = split_group_olfactory(odp_subjects)

            # 合并每个类别的划分结果
            S_train = np.concatenate([ctl_train, odn_train, odp_train])
            S_val = np.concatenate([ctl_val, odn_val, odp_val])
            S_test = np.concatenate([ctl_test, odn_test, odp_test])

            # 保存划分
            self.save_split({"train_subjects": S_train, "val_subjects": S_val, "test_subjects": S_test})
            return S_train, S_val, S_test

        elif self.hparams.dataset_name == 'taowu':
            # 根据类别分组
            control_subjects = [subj for subj in S.keys() if "sub-control" in subj]  # 控制组
            patient_subjects = [subj for subj in S.keys() if "sub-patient" in subj]  # 患者组

            # 定义划分函数
            def split_group_taowu(subjects):
                n_train = int(len(subjects) * self.hparams.train_split)
                n_val = int(len(subjects) * self.hparams.val_split)

                train = np.random.choice(subjects, n_train, replace=False)
                remaining = np.setdiff1d(subjects, train)

                val = np.random.choice(remaining, n_val, replace=False)
                test = np.setdiff1d(subjects, np.concatenate([train, val]))

                return train, val, test

            # 划分两类数据
            control_train, control_val, control_test = split_group_taowu(control_subjects)
            patient_train, patient_val, patient_test = split_group_taowu(patient_subjects)

            # 合并两类的划分结果
            S_train = np.concatenate([control_train, patient_train])
            S_val = np.concatenate([control_val, patient_val])
            S_test = np.concatenate([control_test, patient_test])

            # 保存划分结果
            self.save_split({"train_subjects": S_train, "val_subjects": S_val, "test_subjects": S_test})

        elif self.hparams.dataset_name == 'ABIDE':
            # 从 CSV 文件中加载受试者信息
            df = pd.read_csv(self.hparams.subject_csv_path)  # 假设 CSV 路径由参数指定

            # 根据诊断组（DX_GROUP）分组
            asd_subjects = df[df['DX_GROUP'] == 1]['SUB_ID'].values  # 自闭症组
            td_subjects = df[df['DX_GROUP'] == 2]['SUB_ID'].values  # 典型发育组

            # 定义划分函数
            def split_group_abide(subjects):
                n_train = int(len(subjects) * self.hparams.train_split)
                n_val = int(len(subjects) * self.hparams.val_split)

                train = np.random.choice(subjects, n_train, replace=False)
                remaining = np.setdiff1d(subjects, train)

                val = np.random.choice(remaining, n_val, replace=False)
                test = np.setdiff1d(subjects, np.concatenate([train, val]))

                return train, val, test

            # 划分两类数据
            asd_train, asd_val, asd_test = split_group_abide(asd_subjects)
            td_train, td_val, td_test = split_group_abide(td_subjects)

            # 合并两类的划分结果
            S_train = np.concatenate([asd_train, td_train])
            S_val = np.concatenate([asd_val, td_val])
            S_test = np.concatenate([asd_test, td_test])

            # 保存划分结果
            self.save_split({"train_subjects": S_train, "val_subjects": S_val, "test_subjects": S_test})

            return S_train, S_val, S_test


    def load_split(self):
        subject_order = open(self.split_file_path, "r").readlines()
        subject_order = [x[:-1] for x in subject_order]
        train_index = np.argmax(["train" in line for line in subject_order])
        val_index = np.argmax(["val" in line for line in subject_order])
        test_index = np.argmax(["test" in line for line in subject_order])
        train_names = subject_order[train_index + 1 : val_index]
        val_names = subject_order[val_index + 1 : test_index]
        test_names = subject_order[test_index + 1 :]
        return train_names, val_names, test_names

    def prepare_data(self):
        # This function is only called at global rank==0
        return

    # filter subjects with metadata and pair subject names with their target values (+ sex)
    def make_subject_dict(self):
        # output: {'subj1':[target1,target2],'subj2':[target1,target2]...}
        img_root = os.path.join(self.hparams.image_path, 'img')
        final_dict = dict()

        if self.hparams.dataset_name == "olfactory":
            # 假设我们不依赖元数据文件，而是直接使用目录名作为标签信息
            subject_list = os.listdir(img_root)

            # 为每个被试者分配一个简单的标签。这里假设标签可以从文件夹名推断
            # 例如假设以 "CTL" 开头的文件夹表示对照组，以 "ODN" 开头的表示某种实验组
            for subject in subject_list:
                if "CTL" in subject:
                    sex = 0  # 假设 "CTL" 组性别为 0
                    target = 0  # 给对照组分配标签 0
                elif "ODP" in subject:
                    sex = 2
                    target = 2
                elif "ODN" in subject:
                    sex = 3  # 假设 "ODN" 组性别为 1
                    target = 3  # 给实验组分配标签 1
                else:
                    continue  # 忽略不符合规则的文件夹名

                # 将性别和标签加入到 final_dict 中
                final_dict[subject] = [sex, target]


        elif self.hparams.dataset_name == "taowu":
            subject_list = os.listdir(img_root)

            for subject in subject_list:
                if "sub-control" in subject:  # 控制组
                    sex = 0  # 假设控制组性别为 0
                    target = 0  # 类别标签 0
                elif "sub-patient" in subject:  # 患者组
                    sex = 1  # 假设患者组性别为 1
                    target = 1  # 类别标签 1
                else:
                    continue  # 忽略不符合规则的文件夹名

                # 将性别和标签加入到 final_dict 中
                final_dict[subject] = [sex, target]

        elif self.hparams.dataset_name == "ABIDE":
            # 获取受试者列表
            subject_list = os.listdir(img_root)

            # 加载 ABIDE 的元数据 CSV
            meta_data = pd.read_csv(self.hparams.subject_csv_path)

            # 确定任务目标列
            if self.hparams.downstream_task == 'sex':
                target_column = 'SEX'
            elif self.hparams.downstream_task == 'diagnosis':
                target_column = 'DX_GROUP'
            else:
                raise ValueError('Unsupported downstream task for ABIDE')

            # 筛选有效数据
            meta_task = meta_data[['SUB_ID', 'SEX', 'DX_GROUP']].dropna()

            # 遍历受试者列表并匹配元数据
            for subject in subject_list:
                # 从文件夹名称中提取受试者 ID
                subject_id = int(''.join(filter(str.isdigit, subject)))

                # 如果受试者存在于元数据中
                if subject_id in meta_task['SUB_ID'].values:
                    sex = meta_task[meta_task["SUB_ID"] == subject_id]["SEX"].values[
                              0] - 1  # SEX: 1=Male, 2=Female -> 0/1
                    target = meta_task[meta_task["SUB_ID"] == subject_id][target_column].values[0]
                    if target_column == 'DX_GROUP':
                        target -= 1  # DX_GROUP: 1=ASD, 2=TD -> 0/1

                    # 将性别和目标标签加入到 final_dict
                    final_dict[subject] = [sex, target]


        elif self.hparams.dataset_name == "S1200":
            subject_list = os.listdir(img_root)
            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "HCP_1200_gender.csv"))
            meta_data_residual = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "HCP_1200_precise_age.csv"))
            meta_data_all = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "HCP_1200_all.csv"))
            if self.hparams.downstream_task == 'sex': task_name = 'Gender'
            elif self.hparams.downstream_task == 'age': task_name = 'age'
            elif self.hparams.downstream_task == 'int_total': task_name = 'CogTotalComp_AgeAdj'
            else: raise NotImplementedError()

            if self.hparams.downstream_task == 'sex':
                meta_task = meta_data[['Subject',task_name]].dropna()
            elif self.hparams.downstream_task == 'age':
                meta_task = meta_data_residual[['subject',task_name,'sex']].dropna()
                #rename column subject to Subject
                meta_task = meta_task.rename(columns={'subject': 'Subject'})
            elif self.hparams.downstream_task == 'int_total':
                meta_task = meta_data[['Subject',task_name,'Gender']].dropna()  
            
            for subject in subject_list:
                if int(subject) in meta_task['Subject'].values:
                    if self.hparams.downstream_task == 'sex':
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]
                        target = 1 if target == "M" else 0
                        sex = target
                    elif self.hparams.downstream_task == 'age':
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]
                        sex = meta_task[meta_task["Subject"]==int(subject)]["sex"].values[0]
                        sex = 1 if sex == "M" else 0
                    elif self.hparams.downstream_task == 'int_total':
                        target = meta_task[meta_task["Subject"]==int(subject)][task_name].values[0]
                        sex = meta_task[meta_task["Subject"]==int(subject)]["Gender"].values[0]
                        sex = 1 if sex == "M" else 0
                    final_dict[subject]=[sex,target]
            
        elif self.hparams.dataset_name == "ABCD":
            subject_list = [subj[4:] for subj in os.listdir(img_root)]
            
            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "ABCD_phenotype_total.csv"))
            if self.hparams.downstream_task == 'sex': task_name = 'sex'
            elif self.hparams.downstream_task == 'age': task_name = 'age'
            elif self.hparams.downstream_task == 'int_total': task_name = 'nihtbx_totalcomp_uncorrected'
            else: raise ValueError('downstream task not supported')
           
            if self.hparams.downstream_task == 'sex':
                meta_task = meta_data[['subjectkey',task_name]].dropna()
            else:
                meta_task = meta_data[['subjectkey',task_name,'sex']].dropna()
            
            for subject in subject_list:
                if subject in meta_task['subjectkey'].values:
                    target = meta_task[meta_task["subjectkey"]==subject][task_name].values[0]
                    sex = meta_task[meta_task["subjectkey"]==subject]["sex"].values[0]
                    final_dict[subject]=[sex,target]
            
        elif self.hparams.dataset_name == "UKB":
            if self.hparams.downstream_task == 'sex': task_name = 'sex'
            elif self.hparams.downstream_task == 'age': task_name = 'age'
            elif self.hparams.downstream_task == 'int_fluid' : task_name = 'fluid'
            else: raise ValueError('downstream task not supported')
                
            meta_data = pd.read_csv(os.path.join(self.hparams.image_path, "metadata", "UKB_phenotype_gps_fluidint.csv"))
            if task_name == 'sex':
                meta_task = meta_data[['eid',task_name]].dropna()
            else:
                meta_task = meta_data[['eid',task_name,'sex']].dropna()

            for subject in os.listdir(img_root):
                if subject.endswith('20227_2_0') and (int(subject[:7]) in meta_task['eid'].values):
                    target = meta_task[meta_task["eid"]==int(subject[:7])][task_name].values[0]
                    sex = meta_task[meta_task["eid"]==int(subject[:7])].values[0]
                    final_dict[str(subject[:7])] = [sex,target]
                else:
                    continue 
        
        return final_dict

    def setup(self, stage=None):
        # this function will be called at each devices
        Dataset = self.get_dataset()
        params = {
                "root": self.hparams.image_path,
                "subject_csv_path": self.hparams.subject_csv_path,
                "sequence_length": self.hparams.sequence_length,
                "contrastive":self.hparams.use_contrastive,
                "contrastive_type":self.hparams.contrastive_type,
                "stride_between_seq": self.hparams.stride_between_seq,
                "stride_within_seq": self.hparams.stride_within_seq,
                "with_voxel_norm": self.hparams.with_voxel_norm,
                "downstream_task": self.hparams.downstream_task,
                "shuffle_time_sequence": self.hparams.shuffle_time_sequence,
                "input_type": self.hparams.input_type,
                "label_scaling_method" : self.hparams.label_scaling_method,
                "dtype":'float16'}
        
        subject_dict = self.make_subject_dict()
        if os.path.exists(self.split_file_path):
            train_names, val_names, test_names = self.load_split()
        else:
            train_names, val_names, test_names = self.determine_split_randomly(subject_dict)
        
        if self.hparams.bad_subj_path:
            bad_subjects = open(self.hparams.bad_subj_path, "r").readlines()
            for bad_subj in bad_subjects:
                bad_subj = bad_subj.strip()
                if bad_subj in list(subject_dict.keys()):
                    print(f'removing bad subject: {bad_subj}')
                    del subject_dict[bad_subj]
        
        if self.hparams.limit_training_samples:
            train_names = np.random.choice(train_names, size=self.hparams.limit_training_samples, replace=False, p=None)
        
        train_dict = {key: subject_dict[key] for key in train_names if key in subject_dict}
        val_dict = {key: subject_dict[key] for key in val_names if key in subject_dict}
        test_dict = {key: subject_dict[key] for key in test_names if key in subject_dict}
        
        self.train_dataset = Dataset(**params,subject_dict=train_dict,use_augmentations=False, train=True)
        # load train mean/std of target labels to val/test dataloader
        self.val_dataset = Dataset(**params,subject_dict=val_dict,use_augmentations=False,train=False) 
        self.test_dataset = Dataset(**params,subject_dict=test_dict,use_augmentations=False,train=False) 
        
        print("number of train_subj:", len(train_dict))
        print("number of val_subj:", len(val_dict))
        print("number of test_subj:", len(test_dict))
        print("length of train_idx:", len(self.train_dataset.data))
        print("length of val_idx:", len(self.val_dataset.data))  
        print("length of test_idx:", len(self.test_dataset.data))
        
        # DistributedSampler is internally called in pl.Trainer
        def get_params(train):
            return {
                "batch_size": self.hparams.batch_size if train else self.hparams.eval_batch_size,
                "num_workers": self.hparams.num_workers,
                "drop_last": True,
                "pin_memory": False,
                "persistent_workers": False if self.hparams.dataset_name == 'Dummy' else (train and (self.hparams.strategy == 'ddp')),
                "shuffle": train
            }
        self.train_loader = DataLoader(self.train_dataset, **get_params(train=True))
        self.val_loader = DataLoader(self.val_dataset, **get_params(train=False))
        self.test_loader = DataLoader(self.test_dataset, **get_params(train=False))
        

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        # return self.val_loader
        # currently returns validation and test set to track them during training
        return [self.val_loader, self.test_loader]

    def test_dataloader(self):
        return self.test_loader

    def predict_dataloader(self):
        return self.test_dataloader()

    @classmethod
    def add_data_specific_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=True, formatter_class=ArgumentDefaultsHelpFormatter)
        group = parser.add_argument_group("DataModule arguments")
        group.add_argument("--dataset_split_num", type=int, default=2024) # dataset split, choose from 1, 2, or 3
        group.add_argument("--label_scaling_method", default="standardization", choices=["minmax", "standardization"], help="label normalization strategy for a regression task (mean and std are automatically calculated using train set)")
        group.add_argument("--image_path", default='D:\Project\Parkinson\SwiFT-main\project\data\\ABIDE', help="path to image datasets preprocessed for SwiFT")
        group.add_argument("--bad_subj_path", default=None, help="path to txt file that contains subjects with bad fMRI quality")
        group.add_argument("--input_type", default="rest",choices=['rest', 'task'],help='refer to datasets.py')
        group.add_argument("--train_split", default=0.7, type=float)
        group.add_argument("--val_split", default=0.15, type=float)
        group.add_argument("--batch_size", type=int, default=8)
        group.add_argument("--eval_batch_size", type=int, default=8)
        group.add_argument("--img_size", nargs="+", default=[64, 64, 64, 236], type=int, help="image size (adjust the fourth dimension according to your --sequence_length argument)")
        group.add_argument("--sequence_length", type=int, default=20)
        group.add_argument("--stride_between_seq", type=int, default=1, help="skip some fMRI volumes between fMRI sub-sequences")
        group.add_argument("--stride_within_seq", type=int, default=1, help="skip some fMRI volumes within fMRI sub-sequences")
        group.add_argument("--num_workers", type=int, default=0)
        group.add_argument("--with_voxel_norm", type=str2bool, default=False)
        group.add_argument("--shuffle_time_sequence", action='store_true')
        group.add_argument("--limit_training_samples", type=int, default=None, help="use if you want to limit training samples")
        return parser
