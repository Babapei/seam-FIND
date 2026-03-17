#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
焊缝寻位 Pipeline 统一推理入口
config 指定模型与 checkpoint，输入点云/深度图，输出 3D 轨迹
"""
import os
import sys
import argparse
import numpy as np
import yaml

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch

from seam_pipeline.models import get_model


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def seam_points_to_trajectory(seam_points):
    """缝点按 y 排序得到 3D 轨迹"""
    if len(seam_points) == 0:
        return np.zeros((0, 3))
    order = np.argsort(seam_points[:, 1])
    return seam_points[order]


def run_pointcloud_model(model_name, ckpt_path, points, device="cpu", num_point=2048):
    """点云模型推理：points (N,3) -> seam_points (M,3) -> trajectory"""
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", {})
    num_point_ckpt = ckpt.get("num_point", num_point)

    if model_name == "pointnet":
        m_cfg = cfg.get("pointnet", {})
        model = get_model("pointnet", num_class=m_cfg.get("num_class", 2), num_point=num_point_ckpt)
    elif model_name == "pointnet2":
        m_cfg = cfg.get("pointnet2", {})
        model = get_model("pointnet2", num_class=m_cfg.get("num_class", 2), num_point=num_point_ckpt)
    elif model_name == "dgcnn":
        m_cfg = cfg.get("dgcnn", {})
        model = get_model("dgcnn", num_class=m_cfg.get("num_class", 2), k=m_cfg.get("k", 20),
                         emb_dims=m_cfg.get("emb_dims", 1024), dropout=m_cfg.get("dropout", 0.5))
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[0] > num_point_ckpt:
        idx = np.random.choice(pts.shape[0], num_point_ckpt, replace=False)
        pts = pts[idx]
    elif pts.shape[0] < num_point_ckpt:
        idx = np.random.choice(pts.shape[0], num_point_ckpt, replace=True)
        pts = pts[idx]

    with torch.no_grad():
        x = torch.from_numpy(pts).unsqueeze(0).to(device)
        logits = model(x)
        pred = logits[0].argmax(1).cpu().numpy()

    seam_mask = pred == 1
    seam_pts = pts[seam_mask]
    return seam_points_to_trajectory(seam_pts)


def run_2d_extractor(extractor_name, depth, intrinsics=None, **kwargs):
    """2D 提取器：depth -> uvd -> 需配合相机内参反投影到 3D"""
    from seam_pipeline.extractors import extract_seam_depth_valley, extract_seam_laser_line
    from seam_localization.camera_utils import unproject_pixel

    if extractor_name == "depth_valley":
        seam_uvd = extract_seam_depth_valley(depth, **kwargs)
    elif extractor_name == "laser_line":
        seam_uvd = extract_seam_laser_line(depth, **kwargs)
    else:
        raise ValueError(f"Unknown extractor: {extractor_name}")

    if intrinsics is None or seam_uvd.shape[0] == 0:
        return np.zeros((0, 3))

    trajectory = []
    for i in range(seam_uvd.shape[0]):
        u, v, d = seam_uvd[i, 0], seam_uvd[i, 1], seam_uvd[i, 2]
        x, y, z = unproject_pixel(u, v, d, intrinsics)
        if np.isfinite(x):
            trajectory.append([x, y, z])
    return np.array(trajectory, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="seam_pipeline/config/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--points", type=str, default=None, help=".npy 点云路径")
    parser.add_argument("--depth", type=str, default=None, help=".npy 深度图路径（2D方法）")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["model"]

    if model_name in ("pointnet", "pointnet2", "dgcnn"):
        ckpt = args.checkpoint
        if not ckpt:
            out_cfg = cfg.get("output", {})
            ckpt_dir = out_cfg.get("checkpoint_dir", "seam_pipeline/checkpoints")
            ckpt = os.path.join(_root, ckpt_dir, out_cfg.get("exp_name", "seam_exp"), f"{model_name}_best.pt")
        if not os.path.isfile(ckpt):
            print("Checkpoint 不存在，请先训练")
            sys.exit(1)

        if args.points:
            points = np.load(args.points)
        else:
            from seam_localization.pointcloud_dataset import generate_one_scene
            points, _ = generate_one_scene(num_points_target=2048, seed=42)
        trajectory = run_pointcloud_model(model_name, ckpt, points, args.device)
    else:
        if args.depth:
            depth = np.load(args.depth)
        else:
            from seam_localization.synthetic_data import make_synthetic_depth_with_seam, default_intrinsics
            depth = make_synthetic_depth_with_seam(240, 320)
        intrinsics = default_intrinsics(320, 240)
        ext_cfg = cfg.get("depth_valley", {}) if model_name == "depth_valley" else cfg.get("laser_line", {})
        trajectory = run_2d_extractor(model_name, depth, intrinsics, **ext_cfg)

    print("Trajectory points:", len(trajectory))
    out_path = args.out or "seam_pipeline/output/trajectory.txt"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savetxt(out_path, trajectory, fmt="%.6f", header="x y z (camera frame)")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
