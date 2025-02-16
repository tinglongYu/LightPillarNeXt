import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_scatter
except Exception as e:
    # Incase someone doesn't want to use dynamic pillar vfe and hasn't installed torch_scatter
    pass

from .vfe_template import VFETemplate

# 通过全连接层将原始特征升维，然后求分组内特征最大值作为组内全局特征，然后拼接到每个点的原始特征中
class PFNLayerV2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        # 标志是否是最后一层。如果是最后一层，则直接输出最大池化结果，不需要特征拼接。
        self.last_vfe = last_layer
        # 是否使用批归一化（BatchNorm）。
        self.use_norm = use_norm

        # 如果不是最后一层，out_channels 会减半，这样后续可以进行特征拼接（拼接后恢复原来的维度）。
        if not self.last_vfe:
            out_channels = out_channels // 2

        # 如果启用了批归一化（use_norm=True），在线性层中禁用偏置项（bias=False），因为偏置项的功能被批归一化取代。
        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        # 如果禁用了批归一化，则启用线性层的偏置项。
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
        
        self.relu = nn.ReLU()
    # inputs (N,in_channels) N个点，每个点的特征维度
    # unq_inv (N,) 分组索引，每个点对应一个分组编号，表示这个点是属于那个体素的
    def forward(self, inputs, unq_inv):

        x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]

        if self.last_vfe:
            return x_max
        else:
            x_concatenated = torch.cat([x, x_max[unq_inv, :]], dim=1)
            return x_concatenated


class DynamicPillarVFE(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        self.scale_xy = grid_size[0] * grid_size[1]
        self.scale_y = grid_size[1]
        
        self.grid_size = torch.tensor(grid_size).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, batch_dict, **kwargs):
        points = batch_dict['points'] # (batch_idx, x, y, z, i, e)
            
        voxels = batch_dict['voxels']  # [N, 5, 4]
        voxel_coords = batch_dict['voxel_coords']  # [N, 4]
        voxel_num_points = batch_dict['voxel_num_points']  # [N,]
        
        # 获取每个体素中有效点的索引
        valid_points_mask = torch.arange(voxels.shape[1], device=voxels.device).repeat(len(voxels), 1) < voxel_num_points.unsqueeze(1)
        valid_voxels = voxels[valid_points_mask]  # 去掉全0的点，保留有效点
        
        # 创建 batch_idx 张量并扩展维度
        batch_idx = voxel_coords[:, 0].unsqueeze(1).repeat(1, valid_voxels.shape[1])
        
        # 获取有效的 x, y, z 坐标
        voxel_xyz = valid_voxels[:, :, :3]
        
        # 合并 batch_idx 和 (x, y, z)
        points = torch.cat([batch_idx.view(-1, 1), voxel_xyz.view(-1, 3)], dim=1)
        
        points_coords = torch.floor((points[:, [1,2]] - self.point_cloud_range[[0,1]]) / self.voxel_size[[0,1]]).int()
        mask = ((points_coords >= 0) & (points_coords < self.grid_size[[0,1]])).all(dim=1)
        points = points[mask]
        points_coords = points_coords[mask]
        points_xyz = points[:, [1, 2, 3]].contiguous()

        merge_coords = points[:, 0].int() * self.scale_xy + \
                       points_coords[:, 0] * self.scale_y + \
                       points_coords[:, 1]
        
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        f_cluster = points_xyz - points_mean[unq_inv, :]

        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        if self.use_absolute_xyz:
            features = [points[:, 1:], f_cluster, f_center]
        else:
            features = [points[:, 4:], f_cluster, f_center]
        
        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)
        
        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)
        # features = self.linear1(features)
        # features_max = torch_scatter.scatter_max(features, unq_inv, dim=0)[0]
        # features = torch.cat([features, features_max[unq_inv, :]], dim=1)
        # features = self.linear2(features)
        # features = torch_scatter.scatter_max(features, unq_inv, dim=0)[0]
        
        # generate voxel coordinates
        unq_coords = unq_coords.int()
        voxel_coords = torch.stack((unq_coords // self.scale_xy,
                                   (unq_coords % self.scale_xy) // self.scale_y,
                                   unq_coords % self.scale_y,
                                   torch.zeros(unq_coords.shape[0]).to(unq_coords.device).int()
                                   ), dim=1)
        voxel_coords = voxel_coords[:, [0, 3, 2, 1]]

        batch_dict['voxel_features'] = batch_dict['pillar_features'] = features
        batch_dict['voxel_coords'] = voxel_coords
        return batch_dict


class LightDynamicPillarVFE2D(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, grid_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        # 是否使用归一化（USE_NORM）。
        self.use_norm = self.model_cfg.USE_NORM
        # 是否添加点与原点的距离特征（WITH_DISTANCE）。
        self.with_distance = self.model_cfg.WITH_DISTANCE
        # 是否包含绝对坐标特征
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        # self.use_cluster_xyz = self.model_cfg.get('USE_CLUSTER_XYZ', True)
        if self.use_absolute_xyz:
            num_point_features += 3
        # if self.use_cluster_xyz:
        #     num_point_features += 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        # PFNLayerV2 是用于点特征提取的核心模块，通常是一个小型神经网络。
        # 每个 PFNLayerV2 连接相邻两层点特征，从低维特征提取高阶表达。
        # self.num_filters中只有一个值，所以只有一个pfn_layer
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayerV2(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # 每个体素在三个维度的大小
        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]

        # 计算体素相对于点云空间最小的位置的偏移量
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

        # 网格在xy平面的分辨率
        self.scale_xy = grid_size[0] * grid_size[1]
        self.scale_y = grid_size[1]

        self.grid_size = torch.tensor(grid_size[:2]).cuda()
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def forward(self, batch_dict, **kwargs):
        voxels = batch_dict['voxels']  # shape: [N, 5, 4]
        voxel_coords = batch_dict['voxel_coords']  # shape: [N, 4]
        
        # Step 1: 将样本编号添加到每个体素的点中
        batch_idx = voxel_coords[:, 0].unsqueeze(1)  # shape: [N, 1]，提取每个体素的样本编号
        batch_idx = batch_idx.expand(-1, 5)  # shape: [N, 5]，广播到每个点
        voxels_with_batch_idx = torch.cat([batch_idx.unsqueeze(2), voxels], dim=2)  # shape: [N, 5, 5]
        
        # Step 2: 去掉所有 x, y, z, 强度 全为0的点
        # Create a mask for non-zero points in the x, y, z, intensity columns
        non_zero_mask = (voxels_with_batch_idx[:, :, 1:] != 0).any(dim=2)  # shape: [N, 5]
        
        # Step 3: 将所有的点合并
        # Flatten the valid points while keeping track of the batch_idx
        points = voxels_with_batch_idx[non_zero_mask]  # shape: [M, 5], M is the number of valid points

        # 将点的二维平面坐标映射到离散的体素网格上，shape为(N, 2)
        points_coords = torch.floor(
            (points[:, [1, 2]] - self.point_cloud_range[[0, 1]]) / self.voxel_size[[0, 1]]).int()
        

        points_xyz = points[:, [1, 2, 3]].contiguous()

        # 根据点的 batch_idx 和体素坐标 (x, y)，为每个点生成一个唯一的标识符 merge_coords
        # merge_coords 为每个点生成了一个唯一的标识符，该标识符基于点的 batch_idx 和体素坐标 (x, y) 进行计算。
        # 这样做的目的是为了能够在后续的处理中区分不同的点，即使它们属于不同的批次或处于不同的体素中。
        merge_coords = points[:, 0].int() * self.scale_xy + \
                       points_coords[:, 0] * self.scale_y + \
                       points_coords[:, 1]
        
        # 找出 merge_coords 中的唯一值，表示每个体素的标识符。
        # 记录每个点所属的体素索引。
        # 统计每个体素中的点数。
        # merge_coords = torch.tensor([6, 22, 18, 34, 6, 22, 6])
        # unq_coords = torch.tensor([6, 22, 18, 34])
        # unq_inv = torch.tensor([0, 1, 2, 3, 0, 1, 0])
        # unq_cnt = torch.tensor([3, 2, 1, 1])
        unq_coords, unq_inv, unq_cnt = torch.unique(merge_coords, return_inverse=True, return_counts=True, dim=0)

        # 计算每个点到其所属体素中心的偏移量 f_center，这有助于后续的点特征学习。
        f_center = torch.zeros_like(points_xyz)
        f_center[:, 0] = points_xyz[:, 0] - (points_coords[:, 0].to(points_xyz.dtype) * self.voxel_x + self.x_offset)
        f_center[:, 1] = points_xyz[:, 1] - (points_coords[:, 1].to(points_xyz.dtype) * self.voxel_y + self.y_offset)
        f_center[:, 2] = points_xyz[:, 2] - self.z_offset

        # 将多种点特征（相对偏移、绝对坐标、点强度、点到原点距离等）拼接成一个完整的点特征向量。
        features = [f_center]
        if self.use_absolute_xyz:
            features.append(points[:, 1:])
        else:
            features.append(points[:, 4:])

        # if self.use_cluster_xyz:
        #     points_mean = torch_scatter.scatter_mean(points_xyz, unq_inv, dim=0)
        #     f_cluster = points_xyz - points_mean[unq_inv, :]
        #     features.append(f_cluster)

        if self.with_distance:
            points_dist = torch.norm(points[:, 1:4], 2, dim=1, keepdim=True)
            features.append(points_dist)
        features = torch.cat(features, dim=-1)

        # 使用 PFNLayerV2 对点特征进行学习。
        # unq_inv 用于将点特征映射到对应的体素内，进行聚合和特征提取。
        for pfn in self.pfn_layers:
            features = pfn(features, unq_inv)

        # generate voxel coordinates
        # 将体素的唯一标识符 unq_coords 解码为 (batch_idx, x, y) 格式的三维坐标。
        unq_coords = unq_coords.int()
        # 分别计算的是batch_idx,y,x
        pillar_coords = torch.stack((unq_coords // self.scale_xy,
                                     (unq_coords % self.scale_xy) // self.scale_y,
                                     unq_coords % self.scale_y,
                                     ), dim=1)
        # 交换y，x  最后是batch_idx,x,y
        pillar_coords = pillar_coords[:, [0, 2, 1]]

        batch_dict['pillar_features'] = features
        batch_dict['pillar_coords'] = pillar_coords
        return batch_dict
