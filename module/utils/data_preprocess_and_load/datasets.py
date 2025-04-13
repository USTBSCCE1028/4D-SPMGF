# 4D_fMRI_Transformer
import os
import torch
from torch.utils.data import Dataset, IterableDataset
import torch.nn.functional as F
# import augmentations #commented out because of cv errors
import pandas as pd
from pathlib import Path
import numpy as np
import nibabel as nb
import nilearn
import random

from itertools import cycle
import glob

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, KBinsDiscretizer

class BaseDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__()      
        self.register_args(**kwargs)
        self.sample_duration = self.sequence_length * self.stride_within_seq
        self.stride = max(round(self.stride_between_seq * self.sample_duration),1)
        self.data = self._set_data(self.root, self.subject_dict)
    
    def register_args(self,**kwargs):
        for name,value in kwargs.items():
            setattr(self,name,value)
        self.kwargs = kwargs
    
    def load_sequence(self, subject_path, start_frame, sample_duration, num_frames=None): 
        if self.contrastive:
            num_frames = len(os.listdir(subject_path)) - 2
            y = []
            load_fnames = [f'frame_{frame}.pt' for frame in range(start_frame, start_frame+sample_duration,self.stride_within_seq)]
            if self.with_voxel_norm:
                load_fnames += ['voxel_mean.pt', 'voxel_std.pt']

            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)
                y_loaded = torch.load(img_path).unsqueeze(0)
                y.append(y_loaded)
            y = torch.cat(y, dim=4)
            
            random_y = []
            
            full_range = np.arange(0, num_frames-sample_duration+1)
            # exclude overlapping sub-sequences within a subject
            exclude_range = np.arange(start_frame-sample_duration, start_frame+sample_duration)
            available_choices = np.setdiff1d(full_range, exclude_range)
            random_start_frame = np.random.choice(available_choices, size=1, replace=False)[0]
            load_fnames = [f'frame_{frame}.pt' for frame in range(random_start_frame, random_start_frame+sample_duration,self.stride_within_seq)]
            if self.with_voxel_norm:
                load_fnames += ['voxel_mean.pt', 'voxel_std.pt']
            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)
                y_loaded = torch.load(img_path).unsqueeze(0)
                random_y.append(y_loaded)
            random_y = torch.cat(random_y, dim=4)
            return (y, random_y)

        else: # without contrastive learning
            y = []
            if self.shuffle_time_sequence: # shuffle whole sequences
                load_fnames = [f'frame_{frame}.pt' for frame in random.sample(list(range(0,num_frames)),sample_duration//self.stride_within_seq)]
            else:
                load_fnames = [f'frame_{frame}.pt' for frame in range(start_frame, start_frame+sample_duration,self.stride_within_seq)]
            
            if self.with_voxel_norm:
                load_fnames += ['voxel_mean.pt', 'voxel_std.pt']
                
            for fname in load_fnames:
                img_path = os.path.join(subject_path, fname)
                y_i = torch.load(img_path).unsqueeze(0)
                y.append(y_i)
            y = torch.cat(y, dim=4)
            return y

    def __len__(self):
        return  len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError("Required function")

    def _set_data(self, root, subject_dict):
        raise NotImplementedError("Required function")

class S1200(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')
        for i, subject in enumerate(subject_dict):
            sex,target = subject_dict[subject]
            subject_path = os.path.join(img_root, subject)
            num_frames = len(os.listdir(subject_path)) - 2 # voxel mean & std
            session_duration = num_frames - self.sample_duration + 1
            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)
        return data

    def __getitem__(self, index):
        _, subject, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        # target = self.label_dict[target] if isinstance(target, str) else target.float()

        if self.contrastive:
            y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            y = torch.nn.functional.pad(y, (8, 7, 2, 1, 11, 10), value=background_value) # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)

            background_value = rand_y.flatten()[0]
            rand_y = rand_y.permute(0,4,1,2,3)
            rand_y = torch.nn.functional.pad(rand_y, (8, 7, 2, 1, 11, 10), value=background_value) # adjust this padding level according to your data
            rand_y = rand_y.permute(0,2,3,4,1)

            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subject,
                "target": target,
                "TR": start_frame,
                "sex": sex
            }

        else:
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            y = torch.nn.functional.pad(y, (8, 7, 2, 1, 11, 10), value=background_value) # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)

            return {
                "fmri_sequence": y,
                "subject_name": subject,
                "target": target,
                "TR": start_frame,
                "sex": sex,
            } 

class ABCD(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            # subject_name = subject[4:]
            
            subject_path = os.path.join(img_root, 'sub-'+subject_name)

            num_frames = len(os.listdir(subject_path)) - 2 # voxel mean & std
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
                        
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        #age = self.label_dict[age] if isinstance(age, str) else age.float()
        
        #contrastive learning
        if self.contrastive:
            y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            # ABCD image shape: 79, 97, 85
            y = torch.nn.functional.pad(y, (6, 5, 0, 0, 9, 8), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)

            background_value = rand_y.flatten()[0]
            rand_y = rand_y.permute(0,4,1,2,3)
            # ABCD image shape: 79, 97, 85
            rand_y = torch.nn.functional.pad(rand_y, (6, 5, 0, 0, 9, 8), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
            rand_y = rand_y.permute(0,2,3,4,1)

            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex
            } 

        # resting or task
        else:   
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            if self.input_type == 'rest':
                # ABCD rest image shape: 79, 97, 85
                # latest version might be 96,96,95
                y = torch.nn.functional.pad(y, (6, 5, 0, 0, 9, 8), value=background_value)[:,:,:,:96,:] # adjust this padding level according to your data
            elif self.input_type == 'task':
                # ABCD task image shape: 96, 96, 95
                # background value = 0
                # minmax scaled in brain (0~1)
                y = torch.nn.functional.pad(y, (0, 1, 0, 0, 0, 0), value=background_value) # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)

            return {
                "fmri_sequence": y,
                "subject_name": subject_name,
                "target": target,
                "TR": start_frame,
                "sex": sex,
            } 

class UKB(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')
        # subject_list = [subj for subj in os.listdir(img_root) if subj.endswith('20227_2_0')] # only use release 2

        for i, subject_name in enumerate(subject_dict):
            sex, target = subject_dict[subject_name]
            subject20227 = str(subject_name)+'_20227_2_0'
            subject_path = os.path.join(img_root, subject20227)
            num_frames = len(os.listdir(subject_path)) - 2 # voxel mean & std
            session_duration = num_frames - self.sample_duration + 1

            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject_name, subject_path, start_frame, self.stride, num_frames, target, sex)
                data.append(data_tuple)
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def __getitem__(self, index):
        _, subject_name, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]
        if self.contrastive:
                y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)

                background_value = y.flatten()[0]
                y = y.permute(0,4,1,2,3)
                y = torch.nn.functional.pad(y, (3, 2, -7, -6, 3, 2), value=background_value) # adjust this padding level according to your data
                y = y.permute(0,2,3,4,1)

                background_value = rand_y.flatten()[0]
                rand_y = rand_y.permute(0,4,1,2,3)
                rand_y = torch.nn.functional.pad(rand_y, (3, 2, -7, -6, 3, 2), value=background_value) # adjust this padding level according to your data
                rand_y = rand_y.permute(0,2,3,4,1)

                return {
                    "fmri_sequence": (y, rand_y),
                    "subject_name": subject_name,
                    "target": target,
                    "TR": start_frame,
                    "sex": sex
                }
        else:
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            background_value = y.flatten()[0]
            y = y.permute(0,4,1,2,3)
            y = torch.nn.functional.pad(y, (3, 2, -7, -6, 3, 2), value=background_value) # adjust this padding level according to your data
            y = y.permute(0,2,3,4,1)
            return {
                        "fmri_sequence": y,
                        "subject_name": subject_name,
                        "target": target,
                        "TR": start_frame,
                        "sex": sex,
                    } 
    
class Dummy(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, total_samples=100000)
        

    def _set_data(self, root, subject_dict):
        data = []
        for k in range(0,self.total_samples):
            data.append((k, 'subj'+ str(k), 'path'+ str(k), self.stride))
        
        # train dataset
        # for regression tasks
        if self.train: 
            self.target_values = np.array([val for val in range(len(data))]).reshape(-1, 1)
            
        return data

    def __len__(self):
        return self.total_samples

    def __getitem__(self,idx):
        _, subj, _, sequence_length = self.data[idx]
        y = torch.randn(( 1, 96, 96, 96, sequence_length),dtype=torch.float16) #self.y[seq_idx]
        sex = torch.randint(0,2,(1,)).float()
        target = torch.randint(0,2,(1,)).float()

        if self.contrastive:
            rand_y = torch.randn(( 1, 96, 96, 96, sequence_length),dtype=torch.float16)
            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subj,
                "target": target,
                "TR": 0,
                }
        else:
            return {
                    "fmri_sequence": y,
                    "subject_name": subj,
                    "target": target,
                    "TR": 0,
                    "sex": sex,
                    }

class OlfactoryDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        # 遍历所有被试者
        for i, subject in enumerate(subject_dict):
            if "CTL" in subject:
                sex = 0  # 假设性别标签，保持原样
                target = 0  # 将CTL标签设为0
            elif "ODP" in subject:
                sex = 1
                target = 1  # 将ODP标签设为1
            elif "ODN" in subject:
                sex = 2
                target = 2  # 将ODN标签设为2
            else:
                continue  # 忽略不符合规则的文件夹名
            subject_path = os.path.join(img_root, subject)
            num_frames = len([f for f in os.listdir(subject_path) if f.startswith('frame_')])  # 只统计frame文件
            session_duration = num_frames - self.sample_duration + 1

            # 创建时间序列
            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame, self.sequence_length, num_frames, target, sex)
                data.append(data_tuple)

        # 如果是训练数据集，并且是回归任务，存储目标值的均值和标准差用于标准化
        if self.train:
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data



    def __getitem__(self, index):
        _, subject, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]

        # 加载时间序列数据
        if self.contrastive:
            # 对比学习模式下加载两个不同的序列
            y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)

            # 保持 batch 维度，调整为 [1, 91, 109, 91, 20]
            background_value = y.flatten()[0]

            # 填充和裁剪到目标尺寸 [1, 96, 96, 96, 20]
            y = F.pad(y, (0, 0, 2, 3, 0, 0, 2, 3), value=background_value)  # 填充 H 和 D 维度
            y = y[:, :96, :96, :, :]  # 裁剪 W 维度到96

            background_value = rand_y.flatten()[0]
            rand_y = F.pad(rand_y, (0, 0, 2, 3, 0, 0, 2, 3), value=background_value)  # 填充 H 和 D 维度
            rand_y = rand_y[:, :96, :96, :, :]  # 裁剪 W 维度到96

            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subject,
                "target": target,
                "TR": start_frame,
                "sex": sex
            }
        else:
            # 非对比学习模式下加载单一序列
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            # 保持 batch 维度，调整为 [1, 91, 109, 91, 20]
            background_value = y.flatten()[0]

            # 填充和裁剪到目标尺寸 [1, 96, 96, 96, 20]
            y = F.pad(y, (0, 0, 2, 3, 0, 0, 2, 3), value=background_value)  # 填充 H 和 D 维度
            y = y[:, :96, :96, :, :]  # 裁剪 W 维度到96

            return {
                "fmri_sequence": y,
                "subject_name": subject,
                "target": target,
                "TR": start_frame,
                "sex": sex
            }

class taowuDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        # 遍历所有被试者
        for i, subject in enumerate(subject_dict):
            if "sub-control" in subject:  # 控制组
                target = 0  # 类别标签 0
                sex = 0     # 性别标签 0
            elif "sub-patient" in subject:  # 患者组
                target = 1  # 类别标签 1
                sex = 1     # 性别标签 1
            else:
                continue  # 忽略不符合规则的文件夹名

            subject_path = os.path.join(img_root, subject)
            num_frames = len([f for f in os.listdir(subject_path) if f.startswith('frame_')])  # 只统计frame文件
            session_duration = num_frames - self.sample_duration + 1

            # 创建时间序列
            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame, self.sequence_length, num_frames, target, sex)
                data.append(data_tuple)

        # 如果是训练数据集，并且是回归任务，存储目标值的均值和标准差用于标准化
        if self.train:
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def __getitem__(self, index):
        _, subject, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]

        # 加载时间序列数据
        if self.contrastive:
            # 对比学习模式下加载两个不同的序列
            y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)

            # 保持 batch 维度，调整为 [1, 91, 109, 91, 20]
            background_value = y.flatten()[0]

            # 填充和裁剪到目标尺寸 [1, 96, 96, 96, 20]
            y = F.pad(y, (0, 0, 2, 3, 0, 0, 2, 3), value=background_value)  # 填充 H 和 D 维度
            y = y[:, :96, :96, :, :]  # 裁剪 W 维度到96

            background_value = rand_y.flatten()[0]
            rand_y = F.pad(rand_y, (0, 0, 2, 3, 0, 0, 2, 3), value=background_value)  # 填充 H 和 D 维度
            rand_y = rand_y[:, :96, :96, :, :]  # 裁剪 W 维度到96

            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subject,
                "target": target,
                "TR": start_frame,
                "sex": sex
            }
        else:
            # 非对比学习模式下加载单一序列
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            # 保持 batch 维度，调整为 [1, 91, 109, 91, 20]
            background_value = y.flatten()[0]

            # 填充和裁剪到目标尺寸 [1, 96, 96, 96, 20]
            y = F.pad(y, (0, 0, 2, 3, 0, 0, 2, 3), value=background_value)  # 填充 H 和 D 维度
            y = y[:, :96, :96, :, :]  # 裁剪 W 维度到96

            return {
                "fmri_sequence": y,
                "subject_name": subject,
                "target": target,
                "TR": start_frame,
                "sex": sex
            }

    '''
    def __getitem__(self, index):
        _, subject, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]

        # 加载时间序列数据
        if self.contrastive:
            # 对比学习模式下加载两个不同的序列
            y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)

            # 填充数据到固定尺寸
            background_value = y.flatten()[0]
            y = y.permute(0, 4, 1, 2, 3)
            y = torch.nn.functional.pad(y, (8, 7, 2, 1, 11, 10), value=background_value)
            y = y.permute(0, 2, 3, 4, 1)

            background_value = rand_y.flatten()[0]
            rand_y = rand_y.permute(0, 4, 1, 2, 3)
            rand_y = torch.nn.functional.pad(rand_y, (8, 7, 2, 1, 11, 10), value=background_value)
            rand_y = rand_y.permute(0, 2, 3, 4, 1)

            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subject,
                "target": target,
                "TR": start_frame,
                "sex": sex
            }
        else:
            # 非对比学习模式下加载单一序列
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            # 填充数据到固定尺寸
            background_value = y.flatten()[0]
            y = y.permute(0, 4, 1, 2, 3)
            y = torch.nn.functional.pad(y, (8, 7, 2, 1, 11, 10), value=background_value)
            y = y.permute(0, 2, 3, 4, 1)

            return {
                "fmri_sequence": y,
                "subject_name": subject,
                "target": target,
                "TR": start_frame,
                "sex": sex
            }
    '''


class ABIDEDataset(BaseDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _set_data(self, root, subject_dict):
        data = []
        img_root = os.path.join(root, 'img')

        # 遍历所有被试者
        for i, (subject, (sex, target)) in enumerate(subject_dict.items()):
            subject_path = os.path.join(img_root, subject)
            target = target -1
            if target == 2:
                print()
            if not os.path.exists(subject_path):
                print(f"Subject path {subject_path} not found. Skipping...")
                continue

            num_frames = len([f for f in os.listdir(subject_path) if f.startswith('frame_')])  # 只统计 frame 文件
            session_duration = num_frames - self.sample_duration + 1

            # 创建时间序列
            for start_frame in range(0, session_duration, self.stride):
                data_tuple = (i, subject, subject_path, start_frame, self.sequence_length, num_frames, target, sex)
                data.append(data_tuple)

        # 如果是训练数据集，并且是回归任务，存储目标值的均值和标准差用于标准化
        if self.train:
            self.target_values = np.array([tup[6] for tup in data]).reshape(-1, 1)

        return data

    def __getitem__(self, index):
        _, subject, subject_path, start_frame, sequence_length, num_frames, target, sex = self.data[index]

        # 加载时间序列数据
        if self.contrastive:
            # 对比学习模式下加载两个不同的序列
            y, rand_y = self.load_sequence(subject_path, start_frame, sequence_length)

            # 获取背景值
            background_value = y.flatten()[0]

            # 第一步：填充第一维度（Z）和第三维度（X）到 64
            y = F.pad(y, (0, 0, 1, 2, 0, 0, 1, 2), value=background_value)  # 填充 Z 和 X 维度
            rand_y = F.pad(rand_y, (0, 0, 1, 2, 0, 0, 1, 2), value=background_value)  # 同样填充

            # 第二步：裁剪第二维度（Y）到 64
            y = y[:, :, :64, :, :]  # 裁剪 Y 维度到 64
            rand_y = rand_y[:, :, :64, :, :]  # 同样裁剪

            return {
                "fmri_sequence": (y, rand_y),
                "subject_name": subject,
                "target": target,
                "TR": start_frame,
                "sex": sex
            }
        else:
            # 非对比学习模式下加载单一序列
            y = self.load_sequence(subject_path, start_frame, sequence_length, num_frames)

            # 获取背景值
            #background_value = y.flatten()[0]

            y = y.permute(0, 4, 2, 3, 1)

            y = F.pad(y, (0, 3), value=0)

            y = y.permute(0, 4, 2, 3, 1)

            y = y.permute(0, 1, 2, 4, 3)

            y = F.pad(y, (0, 3), value=0)

            y = y.permute(0, 1, 2, 4, 3)


            # 第一步：填充第一维度（Z）和第三维度（X）到 64
            #y = F.pad(y, (0, 0, 1, 2, 0, 0, 1, 2), value=0)  # 填充 Z 和 X 维度

            # 第二步：裁剪第二维度（Y）到 64
            y = y[:, :, :64, :, :]  # 裁剪 Y 维度到 64

            return {
                "fmri_sequence": y,
                "subject_name": subject,
                "target": target,
                "TR": start_frame,
                "sex": sex
            }