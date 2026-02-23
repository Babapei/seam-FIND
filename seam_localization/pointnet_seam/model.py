"""
PointNet 语义分割（二类：缝=1 / 背景=0），PyTorch 实现。
输入 (B, N, 3)，输出 (B, N, 2) logits。
"""
import torch
import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetSeg(nn.Module):
    def __init__(self, num_class=2):
        super(PointNetSeg, self).__init__()
        # 不需要传入 num_point，我们要动态适应
        self.num_class = num_class
        
        # 1. 提取特征 (升维)
        # 3 -> 64 -> 128 -> 1024 (加深网络，提取更强全局特征)
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1) # 把全局特征拉大到 1024
        self.bn3 = nn.BatchNorm1d(1024)
        
        # 2. 分割网络 (降维)
        # 输入维度 = 局部特征(128) + 全局特征(1024) = 1152
        self.conv4 = nn.Conv1d(1152, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)
        self.conv5 = nn.Conv1d(512, 256, 1)
        self.bn5 = nn.BatchNorm1d(256)
        self.conv6 = nn.Conv1d(256, num_class, 1)

    def forward(self, x):
        # x shape: [Batch, Points, 3] -> 需要转置为 [Batch, 3, Points]
        x = x.transpose(2, 1) 
        n_pts = x.size(2) # 动态获取点的数量，不要写死
        
        # --- 阶段1: 提取局部与全局特征 ---
        x = F.relu(self.bn1(self.conv1(x)))
        point_feat = F.relu(self.bn2(self.conv2(x))) # [B, 128, N] -> 记住这个局部特征
        
        x = F.relu(self.bn3(self.conv3(point_feat))) # [B, 1024, N]
        global_feat = torch.max(x, 2, keepdim=True)[0] # [B, 1024, 1] -> 全局最大池化
        
        # --- 阶段2: 拼接 ---
        # 把全局特征复制 N 份
        global_feat_repeat = global_feat.repeat(1, 1, n_pts) # [B, 1024, N]
        # 拼起来: 128(Local) + 1024(Global) = 1152
        concat = torch.cat([point_feat, global_feat_repeat], 1) 
        
        # --- 阶段3: 逐点分类 ---
        x = F.relu(self.bn4(self.conv4(concat)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x) # 输出 Logits, 不需要 Softmax (Loss函数里自带)
        
        # 转回 [Batch, Points, Class] 以便计算 Loss
        return x.transpose(2, 1)

# 测试一下
if __name__ == '__main__':
    sim_data = torch.rand(8, 2048, 3) # Batch=8, Points=2048, XYZ=3
    model = PointNetSeg(num_class=2)
    output = model(sim_data)
    print("Output Shape:", output.shape) # 应该是 [8, 2048, 2]