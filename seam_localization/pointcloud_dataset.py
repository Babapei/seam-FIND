"""
焊缝点云数据生成：从合成深度图得到点云，并为每个点打上二类标签（缝=1 / 背景=0）。
用于训练 PointNet 等点云分割模型；仿真/真机未就绪时用合成数据。
"""
import numpy as np
import os
from typing import Tuple, Optional

# 使用项目内已有模块
import sys
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)

from seam_localization.camera_utils import CameraIntrinsics, depth_to_point_cloud
from seam_localization.synthetic_data import (
    make_synthetic_depth_with_seam,
    default_intrinsics,
)


def label_seam_points_by_x(
    points: np.ndarray,
    seam_x_threshold: float = 0.015,
) -> np.ndarray:
    """
    根据相机系 x 坐标判断是否在缝上（合成数据中缝在图像中心，反投影后 x≈0）。
    points: (N, 3) 相机系 x,y,z
    返回: (N,) 0=背景, 1=缝
    """
    labels = (np.abs(points[:, 0]) <= seam_x_threshold).astype(np.int64)
    return labels


def generate_one_scene(
    height: int = 240,
    width: int = 320,
    num_points_target: int = 2048,
    seam_x_threshold: float = 0.015,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成一帧：深度图 -> 点云 -> 下采样/上采样到 num_points_target -> 二类标签。
    返回: points (P, 3), labels (P,) 0/1
    """
    if seed is not None:
        np.random.seed(seed)
    depth = make_synthetic_depth_with_seam(
        height=height,
        width=width,
        base_depth=1.0,
        seam_depth_delta=-0.01,
        seam_width_px=4.0,
        noise_std=0.001,
        invalid_ratio=0.0,
    )
    intrinsics = default_intrinsics(width=width, height=height)
    points, _ = depth_to_point_cloud(depth, intrinsics, depth_scale=1.0)
    if points.shape[0] == 0:
        return np.zeros((num_points_target, 3), dtype=np.float32), np.zeros(num_points_target, dtype=np.int64)
    labels = label_seam_points_by_x(points, seam_x_threshold=seam_x_threshold)
    N = points.shape[0]
    if N >= num_points_target:
        idx = np.random.choice(N, num_points_target, replace=False)
    else:
        idx = np.random.choice(N, num_points_target, replace=True)
    return points[idx].astype(np.float32), labels[idx]


def generate_dataset(
    num_train: int = 200,
    num_val: int = 50,
    num_points: int = 2048,
    output_dir: Optional[str] = None,
) -> Tuple[str, str]:
    """
    生成训练/验证集，保存为 .npy。
    返回: (train_npy_path, val_npy_path)，每个 npy 为 dict: 'points' (N,P,3), 'labels' (N,P)
    """
    if output_dir is None:
        output_dir = os.path.join(_here, "pointcloud_data")
    os.makedirs(output_dir, exist_ok=True)
    train_points_list = []
    train_labels_list = []
    for i in range(num_train):
        pts, lab = generate_one_scene(num_points_target=num_points, seed=i)
        train_points_list.append(pts)
        train_labels_list.append(lab)
    train_data = {
        "points": np.stack(train_points_list, axis=0),
        "labels": np.stack(train_labels_list, axis=0),
    }
    train_path = os.path.join(output_dir, "seam_train.npy")
    np.save(train_path, train_data)

    val_points_list = []
    val_labels_list = []
    for i in range(num_val):
        pts, lab = generate_one_scene(num_points_target=num_points, seed=num_train + i)
        val_points_list.append(pts)
        val_labels_list.append(lab)
    val_data = {
        "points": np.stack(val_points_list, axis=0),
        "labels": np.stack(val_labels_list, axis=0),
    }
    val_path = os.path.join(output_dir, "seam_val.npy")
    np.save(val_path, val_data)
    return train_path, val_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_train", type=int, default=200)
    parser.add_argument("--num_val", type=int, default=50)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()
    t, v = generate_dataset(args.num_train, args.num_val, args.num_points, args.out_dir)
    print("Train:", t)
    print("Val:", v)


def get_default_data_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "pointcloud_data")
