"""加载 seam_localization 生成的点云分割数据 (.npy)。"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset

_here = os.path.dirname(os.path.abspath(__file__))
_default_data_dir = os.path.join(os.path.dirname(_here), "pointcloud_data")


class SeamPointCloudDataset(Dataset):
    def __init__(self, npy_path, num_points=None, train=True):
        """
        npy_path: 指向 seam_train.npy 或 seam_val.npy
        每个样本: points (P,3), labels (P,)
        """
        data = np.load(npy_path, allow_pickle=True).item()
        self.points = data["points"]   # (N, P, 3)
        self.labels = data["labels"]   # (N, P)
        self.num_points = num_points or self.points.shape[1]
        self.train = train

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        pts = self.points[idx]
        lab = self.labels[idx]
        if self.train and pts.shape[0] > self.num_points:
            idx_ = np.random.choice(pts.shape[0], self.num_points, replace=False)
            pts = pts[idx_]
            lab = lab[idx_]
        elif pts.shape[0] < self.num_points:
            idx_ = np.random.choice(pts.shape[0], self.num_points, replace=True)
            pts = pts[idx_]
            lab = lab[idx_]
        if self.train:
            # 轻微抖动
            pts = pts + np.random.randn(*pts.shape).astype(np.float32) * 0.005
        return torch.from_numpy(pts), torch.from_numpy(lab).long()
