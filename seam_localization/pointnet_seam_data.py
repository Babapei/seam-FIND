# 加载 pointcloud_dataset 生成的 .npy 用于 PointNet 训练
import os
import numpy as np
import torch
from torch.utils.data import Dataset

_here = os.path.dirname(os.path.abspath(__file__))
_default_data_dir = os.path.join(_here, "pointcloud_data")


class SeamPointCloudDataset(Dataset):
    def __init__(self, npy_path, num_points=None, train=True):
        data = np.load(npy_path, allow_pickle=True).item()
        self.points = data["points"]
        self.labels = data["labels"]
        self.num_points = num_points or self.points.shape[1]
        self.train = train

    def __len__(self):
        return self.points.shape[0]

    def __getitem__(self, idx):
        pts = self.points[idx].copy()
        lab = self.labels[idx].copy()
        if self.train and pts.shape[0] > self.num_points:
            idx_ = np.random.choice(pts.shape[0], self.num_points, replace=False)
            pts, lab = pts[idx_], lab[idx_]
        elif pts.shape[0] < self.num_points:
            idx_ = np.random.choice(pts.shape[0], self.num_points, replace=True)
            pts, lab = pts[idx_], lab[idx_]
        if self.train:
            pts = pts + np.random.randn(*pts.shape).astype(np.float32) * 0.005
        return torch.from_numpy(pts), torch.from_numpy(lab).long()
