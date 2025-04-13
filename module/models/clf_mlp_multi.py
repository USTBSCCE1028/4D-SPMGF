import torch
import torch.nn as nn
import numpy as np
import nibabel as nib


class FMRIToROI(nn.Module):
    def __init__(self, input_dim=192*2*2*2, output_dim=61*73*61, aal_template_path=None, num_classes=2, num_tokens = 192):
        """
        初始化全连接层和模板路径
        :param input_dim: 输入特征维度 (flattened: channels * d1 * d2 * d3)
        :param output_dim: 输出空间大小 (flattened: 61 * 73 * 61)
        :param aal_template_path: AAL 模板文件路径
        """
        super(FMRIToROI, self).__init__()

        num_outputs = 1 if num_classes == 2 else num_classes
        self.hidden = nn.Linear(num_tokens+116, 768)
        # self.head = nn.Linear(4*num_tokens, num_outputs)
        self.out = nn.Linear(768, 1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)


        self.fc = nn.Linear(input_dim, output_dim)
        self.fc1 = nn.Linear(116*116, 64) #64是因为batch为8, 8*8=64
        self.fc2 = nn.Linear(116, 8)
        self.aal_template_path = aal_template_path
        self.fs = nn.ModuleList()
        self.normalize_m = 1

        # 加载 AAL 模板并提取 ROI 体素坐标
        if aal_template_path is not None:
            aal_img = nib.load(aal_template_path)
            self.aal_data = aal_img.get_fdata()
            self.roi_indices = self._extract_roi_indices()
        else:
            raise ValueError("AAL template path must be provided.")

        self.m = [nn.Linear(1, 64, bias=1), nn.ReLU()]
        for _ in range(1, 3 - 1):
            self.m.append(nn.Linear(64, 64, bias=1))
            self.m.append(nn.ReLU())
        self.m.append(nn.Linear(64, 1, bias=1))

        self.m = nn.Sequential(*self.m)

        for _ in range(116):
            curr_f = [nn.Linear(1, 64, bias=1), nn.ReLU(), nn.Dropout(p=0.6)]
            for _ in range(1, 3 - 1):
                curr_f.append(nn.Linear(64, 64, bias=1))
                curr_f.append(nn.ReLU())
                curr_f.append(nn.Dropout(p=0.6))
            curr_f.append(nn.Linear(64, 1, bias=1))
            self.fs.append(nn.Sequential(*curr_f))

    def _extract_roi_indices(self):
        """
        提取每个 ROI 的体素坐标
        :return: 字典，每个 ROI 分区编号对应一个体素坐标列表
        """
        roi_indices = {}
        for roi in range(1, 117):  # ROI 编号从 1 到 116
            coords = np.array(np.where(self.aal_data == roi)).T  # 提取该 ROI 的所有体素坐标
            if coords.size > 0:
                roi_indices[roi] = coords
        return roi_indices

    def forward(self, x):
        """
        前向传播，将输入 Tensor 转换为 ROI 时间序列
        :param x: 输入 Tensor，形状为 [batch_size, channels, d1, d2, d3, time_dim]
        :return: 节点特征 (x)，边信息 (edge_index)，和节点距离 (node_distances)
        """
        device = x.device

        batch_size, channels, d1, d2, d3, time_dim = x.shape

        # Flatten 空间维度
        x_flattened = x.view(batch_size, -1, time_dim)  # [batch_size, 1536, time_dim]

        # 全连接层映射
        x_fc = self.fc(x_flattened.permute(0, 2, 1))  # [batch_size, time_dim, 1536] -> [batch_size, time_dim, 271633]
        x_fc = x_fc.permute(0, 2, 1)  # [batch_size, time_dim, 271633] -> [batch_size, 271633, time_dim]

        # Reshape 到目标空间维度
        x_reshaped = x_fc.view(batch_size, 61, 73, 61, time_dim)  # [batch_size, 61, 73, 61, time_dim]

        # 逐个 Batch 提取 ROI 时间序列
        roi_time_series_list = []
        for batch_index in range(batch_size):
            # 提取当前 Batch 的数据
            fmri_data = x_reshaped[batch_index].detach().cpu().numpy()  # [61, 73, 61, time_dim]

            # 初始化当前 Batch 的 ROI 时间序列
            time_series = np.zeros((time_dim, 116))  # [time_dim, 116]

            # 遍历每个 ROI 提取时间序列
            for roi, coords in self.roi_indices.items():
                # 根据 ROI 体素坐标提取时间序列
                roi_voxels = np.array([fmri_data[tuple(coord)] for coord in coords])
                time_series[:, roi - 1] = roi_voxels.mean(axis=0)  # 每个时间点取平均值

            roi_time_series_list.append(torch.tensor(time_series, dtype=torch.float32))  # 转为 Tensor

        # 堆叠所有 Batch 的 ROI 时间序列
        roi_time_series_tensor = torch.stack(roi_time_series_list, dim=0)  # [batch_size, time_dim, 116]

        # 节点特征 (x)
        x_node = torch.mean(roi_time_series_tensor, dim=1)  # [batch_size, 116]

        # 使用独立函数计算边信息和节点距离
        edge_index = compute_edge_info(roi_time_series_tensor)
        node_distances = compute_node_distances(self.roi_indices)

        ##############ROI
        x_node = x_node.to(device)
        edge_index = edge_index.to(device)
        node_distances = node_distances.to(device)
        fx = torch.empty(x_node.size(0), x_node.size(1), 1).to(device)
        for feature_index in range(x_node.size(1)):
            feature_col = x_node[:, feature_index]
            feature_col = feature_col.view(-1, 1)
            feature_col = self.fs[feature_index](feature_col)
            fx[:, feature_index] = feature_col

        fx_perm = torch.permute(fx, (2, 0, 1))

        #if self.normalize_m:
        #    node_distances = torch.div(node_distances, inputs.normalization_matrix)

        m_dist = self.m(node_distances.flatten().view(-1, 1)).view(x_node.size(1), x_node.size(1), 1)

        #调整为batch（8）形状
        m_dist = m_dist.view(-1)
        m_dist = self.fc1(m_dist)
        m_dist = m_dist.view(batch_size, batch_size, 1)
        m_dist_perm = torch.permute(m_dist, (2, 0, 1))
        mf = torch.matmul(m_dist_perm, fx_perm)
        mf = mf.squeeze(0)
        # hidden = torch.sum(mf, dim=1)
        # out = self.fc2(hidden)
        #out = torch.sum(hidden, dim=1).view(batch_size, 1)

        ############fMRI
        x = x.flatten(start_dim=2).transpose(1, 2)  # B L C
        # x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)

        ############out
        out = torch.cat((mf, x), dim=1)
        out = self.hidden(out)
        out = self.out(out)

        # x = self.hidden(x)
        # x = self.head(x)

        return out


def compute_edge_info(roi_time_series_tensor):
    """
    计算 ROI 时间序列之间的皮尔逊相关系数，生成功能连接图。
    :param roi_time_series_tensor: Tensor，形状为 [batch_size, time_dim, 116]
    :return: 边索引 Tensor，形状为 [2, num_edges]
    """
    batch_size, time_dim, num_rois = roi_time_series_tensor.shape
    functional_connectivity = []

    for batch_idx in range(batch_size):
        # 当前 batch 的时间序列
        time_series = roi_time_series_tensor[batch_idx].detach().cpu().numpy()  # [time_dim, 116]

        # 计算皮尔逊相关系数矩阵
        corr_matrix = np.corrcoef(time_series.T)  # [116, 116]
        functional_connectivity.append(corr_matrix)

    # 取第一个 batch 的功能连接图（假设 batch 之间共享边信息）
    corr_matrix = functional_connectivity[0]
    threshold = 0.5  # 设置相关性阈值，过滤弱连接
    edge_index = np.array(np.where(corr_matrix > threshold))  # 边列表，形状为 [2, num_edges]
    return torch.tensor(edge_index, dtype=torch.long)  # 转为 Tensor


def compute_node_distances(roi_indices):
    """
    根据 AAL 模板中 ROI 的体素坐标计算节点之间的物理距离。
    :param roi_indices: 字典，每个 ROI 编号对应的体素坐标列表。
    :return: 节点距离矩阵，形状为 [116, 116]
    """
    num_rois = len(roi_indices)
    roi_centroids = {}

    # 计算每个 ROI 的质心坐标
    for roi, coords in roi_indices.items():
        roi_centroids[roi] = np.mean(coords, axis=0)  # [3]

    # 初始化距离矩阵
    distance_matrix = np.zeros((num_rois, num_rois))

    # 计算每对 ROI 的欧几里得距离
    for i in range(1, num_rois + 1):
        for j in range(1, num_rois + 1):
            distance_matrix[i - 1, j - 1] = np.linalg.norm(roi_centroids[i] - roi_centroids[j])

    return torch.tensor(distance_matrix, dtype=torch.float32)  # [116, 116]