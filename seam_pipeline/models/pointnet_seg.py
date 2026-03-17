# PointNet 二类分割（缝 vs 背景），PyTorch，输入 (B,N,3) 输出 (B,N,num_class)
# 参考 pointnet-master sem_seg 结构，不简化
import torch
import torch.nn as nn


class PointNetSeg(nn.Module):
    def __init__(self, num_class=2, num_point=2048):
        super(PointNetSeg, self).__init__()
        self.num_class = num_class
        self.num_point = num_point
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(256, 128, 1)
        self.conv4 = nn.Conv1d(128, num_class, 1)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = x.transpose(2, 1)  # (B,3,N)
        x = torch.relu(self.bn1(self.conv1(x)))
        point_feat = torch.relu(self.bn2(self.conv2(x)))
        global_feat = torch.max(point_feat, 2, keepdim=True)[0]
        global_feat = global_feat.repeat(1, 1, point_feat.size(2))
        concat = torch.cat([point_feat, global_feat], 1)
        x = torch.relu(self.bn3(self.conv3(concat)))
        x = self.conv4(x)
        return x.transpose(2, 1)  # (B,N,num_class)
