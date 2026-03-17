# PointNet++ 二类分割，PyTorch 纯实现（无自定义 CUDA），参考 pointnet2-master sem_seg
# 使用 KNN 近似 ball query，farthest_point_sample 用循环实现
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def farthest_point_sample(xyz, npoint):
    """B,N,3 -> B,npoint 索引"""
    B, N, _ = xyz.shape
    device = xyz.device
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    batch_indices = torch.arange(B, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    """B,N,C 和 B,S -> B,S,C"""
    B = points.shape[0]
    view_shape = list(idx.shape) + [1]
    repeat_shape = [1] * len(idx.shape) + [points.shape[-1]]
    batch_indices = torch.arange(B, device=points.device).view(B, 1).repeat(1, idx.shape[1])
    new_points = points[batch_indices, idx, :]
    return new_points


def square_distance(src, dst):
    """B,N,C 与 B,M,C -> B,N,M"""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def query_ball_point(radius, nsample, xyz, new_xyz):
    """KNN 近似 ball query"""
    B, N, C = xyz.shape
    S = new_xyz.shape[1]
    dists = square_distance(xyz, new_xyz)
    _, idx = dists.sort(dim=1)
    idx = idx[:, :nsample, :]
    return idx.transpose(1, 2)


def group_points(points, idx):
    """B,N,C 和 B,S,K -> B,S,K,C"""
    B = points.shape[0]
    view_shape = [B, 1, 1] + list(points.shape[2:])
    repeat_shape = [1, idx.shape[1], idx.shape[2]] + [1] * (points.dim() - 2)
    batch_indices = torch.arange(B, device=points.device).view(B, 1, 1).repeat(1, idx.shape[1], idx.shape[2])
    new_points = points[batch_indices, idx, :]
    return new_points


class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super(SetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        B, N, _ = xyz.shape
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)
        idx = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        grouped_xyz = group_points(xyz, idx)
        grouped_xyz -= new_xyz.unsqueeze(2)
        if points is not None:
            grouped_points = group_points(points, idx)
            new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
        else:
            new_points = grouped_xyz
        new_points = new_points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        new_points = torch.max(new_points, 2)[0]
        return new_xyz, new_points


def three_nn(xyz1, xyz2):
    """B,N1,3 与 B,N2,3 -> B,N1,3 最近3点索引, B,N1,3 距离"""
    dists = square_distance(xyz1, xyz2)
    dists, idx = torch.topk(dists, 3, dim=-1, largest=False)
    return idx, dists


def three_interpolate(points, idx, weight):
    """B,N2,C 与 B,N1,3 与 B,N1,3 -> B,N1,C. weight 为 1/distance"""
    B, N2, C = points.shape
    N1 = idx.shape[1]
    dist_recip = weight
    norm = torch.sum(dist_recip, dim=2, keepdim=True).clamp(min=1e-8)
    weight_norm = dist_recip / norm
    weighted_points = torch.zeros(B, N1, C, device=points.device, dtype=points.dtype)
    for i in range(3):
        idx_flat = idx[:, :, i].reshape(B, -1)
        batch_idx = torch.arange(B, device=points.device).view(B, 1).expand(B, N1)
        pts = points[batch_idx, idx_flat.reshape(B, N1), :]
        w = weight_norm[:, :, i:i+1]
        weighted_points += w * pts
    return weighted_points


class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(FeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        B, N, _ = xyz1.shape
        idx, dists = three_nn(xyz1, xyz2)
        dists = dists.clamp(min=1e-10)
        weight = 1.0 / dists
        interpolated_points = three_interpolate(points2, idx, weight)
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points
        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points.permute(0, 2, 1)


class PointNet2Seg(nn.Module):
    """PointNet++ 语义分割，参考 pointnet2-master sem_seg 结构"""

    def __init__(self, num_class=2, num_point=2048):
        super(PointNet2Seg, self).__init__()
        self.sa1 = SetAbstraction(1024, 0.1, 32, 3, [32, 32, 64])
        self.sa2 = SetAbstraction(256, 0.2, 32, 64, [64, 64, 128])
        self.sa3 = SetAbstraction(64, 0.4, 32, 128, [128, 128, 256])
        self.sa4 = SetAbstraction(16, 0.8, 32, 256, [256, 256, 512])
        self.fp4 = FeaturePropagation(768, [256, 256])
        self.fp3 = FeaturePropagation(512, [256, 256])
        self.fp2 = FeaturePropagation(384, [256, 128])
        self.fp1 = FeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.dp1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_class, 1)

    def forward(self, xyz):
        l0_xyz = xyz
        l0_points = None
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)
        x = F.relu(self.bn1(self.conv1(l0_points.permute(0, 2, 1))))
        x = self.dp1(x)
        x = self.conv2(x)
        return x.permute(0, 2, 1)
