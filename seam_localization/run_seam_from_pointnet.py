#!/usr/bin/env python3
# 用训练好的 PointNet 从点云预测焊缝点，并输出 3D 轨迹（缝点按 y 排序后作为轨迹）
import os
import sys
import argparse

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)

import numpy as np
import torch

from seam_localization.pointnet_seam_model import PointNetSeg


def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    num_point = ckpt.get("num_point", 2048)
    model = PointNetSeg(num_class=2, num_point=num_point)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def predict_seam_points(points, model, device="cpu", num_point=2048):
    """
    points: (N, 3) 单帧点云；若 N != num_point 会随机采样到 num_point
    返回: seam_points (M, 3)，预测为缝的点（为送入模型的那批点中的子集）
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] > num_point:
        idx = np.random.choice(pts.shape[0], num_point, replace=False)
        pts = pts[idx]
    elif pts.shape[0] < num_point:
        idx = np.random.choice(pts.shape[0], num_point, replace=True)
        pts = pts[idx]
    pts_batch = torch.from_numpy(pts).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(pts_batch)
    pred = logits[0].argmax(1).cpu().numpy()
    seam_mask = pred == 1
    return pts_batch[0].cpu().numpy()[seam_mask]


def seam_points_to_trajectory(seam_points):
    """缝点按 y 排序后作为 3D 轨迹（相机系）。"""
    if len(seam_points) == 0:
        return np.zeros((0, 3))
    order = np.argsort(seam_points[:, 1])
    return seam_points[order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--points", type=str, default=None, help=".npy 点云 (N,3)，不指定则用合成一帧")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.ckpt is None:
        args.ckpt = os.path.join(_here, "checkpoints", "pointnet_seam.pt")
    if not os.path.isfile(args.ckpt):
        print("未找到模型，请先训练: python seam_localization/train_pointnet_seam.py")
        sys.exit(1)

    model = load_model(args.ckpt, args.device)
    num_point = 2048

    if args.points and os.path.isfile(args.points):
        points = np.load(args.points)
    else:
        from seam_localization.pointcloud_dataset import generate_one_scene
        points, _ = generate_one_scene(num_points_target=num_point, seed=999)
    seam_pts = predict_seam_points(points, model, args.device, num_point)
    trajectory = seam_points_to_trajectory(seam_pts)
    print("Seam points:", len(seam_pts), "Trajectory length:", len(trajectory))

    if args.out:
        np.savetxt(args.out, trajectory, fmt="%.6f", header="x y z")
        print("Saved:", args.out)
    else:
        out_dir = os.path.join(_here, "output")
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, "seam_trajectory_pointnet.txt")
        np.savetxt(p, trajectory, fmt="%.6f", header="x y z")
        print("Saved:", p)


if __name__ == "__main__":
    main()
