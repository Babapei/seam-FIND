"""焊缝点云数据集，从 .npy 加载 (points, labels)"""
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)


class SeamDataset(Dataset):
    """points (N,P,3), labels (N,P) 0=背景 1=缝"""

    def __init__(self, npy_path, num_points=2048):
        data = np.load(npy_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.ndim == 0:
            data = data.item()
        self.points = data["points"]
        self.labels = data["labels"]
        self.num_points = num_points

    def __len__(self):
        return len(self.points)

    def __getitem__(self, i):
        pts = self.points[i].astype(np.float32)
        lab = self.labels[i]
        N = pts.shape[0]
        if N >= self.num_points:
            idx = np.random.choice(N, self.num_points, replace=False)
        else:
            idx = np.random.choice(N, self.num_points, replace=True)
        return torch.from_numpy(pts[idx]), torch.from_numpy(lab[idx]).long()
