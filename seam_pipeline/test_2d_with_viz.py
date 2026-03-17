#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
seam_pipeline 2D 方法测试脚本：生成合成深度图 → 调用 depth_valley/laser_line 提取器 → 输出指标 + 可视化
独立于 test_with_viz.py（深度学习），专门用于 depth_valley、laser_line 两种 2D 传统方法。
"""
import os
import sys
import argparse
import numpy as np
import yaml

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from seam_pipeline.inference import run_2d_extractor
from seam_localization.synthetic_data import (
    make_synthetic_depth_with_seam,
    default_intrinsics,
)
from seam_localization.camera_utils import unproject_pixel


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_gt_trajectory_from_synthetic_depth(depth, intrinsics):
    """
    合成深度图的 GT 轨迹：缝在图像中心，每行取中心列深度反投影到 3D。
    返回 (N, 3)，按 y 排序。
    """
    h, w = depth.shape
    uc = w / 2.0
    traj = []
    for v in range(h):
        col = min(int(round(uc)), w - 1)
        d = depth[v, col]
        if d > 0 and np.isfinite(d):
            x, y, z = unproject_pixel(uc, float(v), d, intrinsics)
            if np.isfinite(x):
                traj.append([x, y, z])
    traj = np.array(traj, dtype=np.float64)
    if traj.shape[0] > 0:
        order = np.argsort(traj[:, 1])
        traj = traj[order]
    return traj


def compute_trajectory_metrics(pred_traj, gt_traj):
    """
    轨迹级指标：pred/gt 互为参考的平均最近邻距离。
    """
    def avg_dist(A, B):
        if A.shape[0] == 0 or B.shape[0] == 0:
            return float("nan")
        # 纯 numpy：每个 A 点到 B 的最小距离
        d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))
        return float(np.mean(d.min(axis=1)))

    d_pred2gt = avg_dist(pred_traj, gt_traj)
    d_gt2pred = avg_dist(gt_traj, pred_traj)
    mean_dist = np.nanmean([d_pred2gt, d_gt2pred]) if not (np.isnan(d_pred2gt) and np.isnan(d_gt2pred)) else float("nan")
    return {"avg_dist_pred2gt_m": d_pred2gt, "avg_dist_gt2pred_m": d_gt2pred, "mean_traj_dist_m": mean_dist}


def visualize_sample_2d(depth, gt_traj_3d, pred_traj_3d, pred_uvd, save_path, sample_id=0):
    """2D 可视化：深度图+GT、深度图+预测、3D 轨迹对比"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm

        for _path in [
            os.path.join(_root, "SimHei.ttf"),
            "/usr/share/fonts/truetype/SimHei.ttf",
        ]:
            if os.path.isfile(_path):
                fm.fontManager.addfont(_path)
                plt.rcParams["font.family"] = fm.FontProperties(fname=_path).get_name()
                break
        plt.rcParams["axes.unicode_minus"] = False
    except ImportError:
        print("未安装 matplotlib，跳过可视化")
        return

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133, projection="3d")

    # 左：深度图 + GT 缝（中心列，绿线）
    ax1.imshow(depth, cmap="gray")
    h, w = depth.shape
    ax1.axvline(x=w / 2, color="lime", linewidth=1.5, label="GT缝(中心)")
    ax1.set_title("Depth + Ground Truth (绿=中心缝)")
    ax1.legend(loc="upper right", fontsize=8)

    # 中：深度图 + 预测缝（红点）
    ax2.imshow(depth, cmap="gray")
    if pred_uvd is not None and pred_uvd.shape[0] > 0:
        ax2.scatter(pred_uvd[:, 0], pred_uvd[:, 1], c="red", s=2, alpha=0.8, label="预测缝")
        ax2.legend(loc="upper right", fontsize=8)
    ax2.set_title("Depth + Prediction (红=预测缝)")

    # 右：3D 轨迹对比，XYZ 等比例显示
    if gt_traj_3d.shape[0] > 0:
        ax3.plot(gt_traj_3d[:, 0], gt_traj_3d[:, 1], gt_traj_3d[:, 2], "b-", linewidth=1.5, label="GT")
    if pred_traj_3d.shape[0] > 0:
        ax3.plot(pred_traj_3d[:, 0], pred_traj_3d[:, 1], pred_traj_3d[:, 2], "r-", linewidth=1, alpha=0.8, label="Pred")
    ax3.set_title("3D 轨迹 (蓝=GT, 红=Pred)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_zlabel("z")
    ax3.legend(loc="upper right", fontsize=8)

    # XYZ 等比例：立方体范围 + set_box_aspect，确保三轴视觉尺度一致
    all_pts = np.vstack([gt_traj_3d, pred_traj_3d]) if (gt_traj_3d.shape[0] > 0 and pred_traj_3d.shape[0] > 0) else (gt_traj_3d if gt_traj_3d.shape[0] > 0 else pred_traj_3d)
    if all_pts.shape[0] > 0:
        mn, mx = all_pts.min(axis=0), all_pts.max(axis=0)
        r = max(mx[0] - mn[0], mx[1] - mn[1], mx[2] - mn[2], 1e-6) / 2
        r = max(r, 0.01)  # 避免过小
        cx, cy, cz = (mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2, (mn[2] + mx[2]) / 2
        ax3.set_box_aspect((1, 1, 1))  # 先设等比例，再设范围
        ax3.set_xlim(cx - r, cx + r)
        ax3.set_ylim(cy - r, cy + r)
        ax3.set_zlim(cz - r, cz + r)

    plt.tight_layout()
    if all_pts.shape[0] > 0:
        fig.canvas.draw()  # 强制应用等比例
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print("  可视化已保存:", save_path)


def run_extractor_and_get_uvd(extractor_name, depth, intrinsics, **kwargs):
    """调用提取器并返回 (trajectory_3d, seam_uvd)"""
    from seam_pipeline.extractors import extract_seam_depth_valley, extract_seam_laser_line

    if extractor_name == "depth_valley":
        seam_uvd = extract_seam_depth_valley(depth, **kwargs)
    elif extractor_name == "laser_line":
        seam_uvd = extract_seam_laser_line(depth, **kwargs)
    else:
        raise ValueError(f"Unknown extractor: {extractor_name}")

    trajectory = run_2d_extractor(extractor_name, depth, intrinsics, **kwargs)
    return trajectory, seam_uvd


def main():
    parser = argparse.ArgumentParser(description="seam_pipeline 2D 方法测试：生成深度图→提取→指标+可视化")
    parser.add_argument("--config", type=str, default="seam_pipeline/config/default.yaml")
    parser.add_argument("--extractor", type=str, default=None, choices=["depth_valley", "laser_line"],
                       help="覆盖 config 中的 model，直接指定 2D 提取器")
    parser.add_argument("--num_test", type=int, default=20, help="测试用例数量")
    parser.add_argument("--num_viz", type=int, default=3, help="可视化样本数量")
    parser.add_argument("--out_dir", type=str, default="seam_pipeline/output/test_2d")
    parser.add_argument("--depth_h", type=int, default=240)
    parser.add_argument("--depth_w", type=int, default=320)
    parser.add_argument("--seed", type=int, default=7000, help="测试集随机种子起始")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path) and not os.path.isfile(config_path):
        config_path = os.path.join(_root, config_path)
    cfg = load_config(config_path)
    extractor_name = args.extractor or cfg["model"]

    if extractor_name not in ("depth_valley", "laser_line"):
        print("test_2d_with_viz 仅支持 depth_valley、laser_line，请修改 config 中的 model")
        sys.exit(1)

    ext_cfg = cfg.get("depth_valley", {}) if extractor_name == "depth_valley" else cfg.get("laser_line", {})
    # laser_line 的 max_line_gap 默认 0.4，相邻像素索引差=1 无法合并，需 >=1 才能与 dense depth 兼容
    if extractor_name == "laser_line":
        ext_cfg = dict(ext_cfg, max_line_gap=max(ext_cfg.get("max_line_gap", 0.4), 2.0))

    intrinsics = default_intrinsics(args.depth_w, args.depth_h)

    print("=== seam_pipeline 2D 测试 ===\n提取器:", extractor_name, "| 测试数:", args.num_test)

    all_metrics = []
    viz_samples = []  # (depth, gt_traj, pred_traj, pred_uvd, i)

    for i in range(args.num_test):
        np.random.seed(args.seed + i)
        depth = make_synthetic_depth_with_seam(
            height=args.depth_h, width=args.depth_w,
            base_depth=1.0, seam_depth_delta=-0.01,
            seam_width_px=4.0, noise_std=0.001, invalid_ratio=0.0,
        )

        gt_traj = get_gt_trajectory_from_synthetic_depth(depth, intrinsics)
        pred_traj, pred_uvd = run_extractor_and_get_uvd(
            extractor_name, depth, intrinsics, **ext_cfg
        )

        m = compute_trajectory_metrics(pred_traj, gt_traj)
        all_metrics.append(m)

        if i < args.num_viz:
            viz_samples.append((depth.copy(), gt_traj, pred_traj, pred_uvd if pred_uvd is not None else np.zeros((0, 3)), i))

    # 汇总指标（过滤全 nan 避免 RuntimeWarning）
    def safe_nanmean(lst):
        valid = [v for v in lst if not np.isnan(v)]
        return float(np.mean(valid)) if valid else float("nan")

    d_pred2gt = safe_nanmean([x["avg_dist_pred2gt_m"] for x in all_metrics])
    d_gt2pred = safe_nanmean([x["avg_dist_gt2pred_m"] for x in all_metrics])
    mean_dist = safe_nanmean([x["mean_traj_dist_m"] for x in all_metrics])

    out_dir = os.path.join(_root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, "test_2d_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"extractor: {extractor_name}\n")
        f.write(f"num_test: {args.num_test}\n")
        f.write(f"avg_dist_pred2gt_m: {d_pred2gt:.6f}\n")
        f.write(f"avg_dist_gt2pred_m: {d_gt2pred:.6f}\n")
        f.write(f"mean_traj_dist_m: {mean_dist:.6f}\n")
    print("\n--- 指标 ---")
    print("avg_dist_pred2gt_m:  {:.6f}".format(d_pred2gt))
    print("avg_dist_gt2pred_m:  {:.6f}".format(d_gt2pred))
    print("mean_traj_dist_m:    {:.6f}".format(mean_dist))
    print("指标已保存:", metrics_path)

    for depth, gt_traj, pred_traj, pred_uvd, i in viz_samples:
        save_path = os.path.join(out_dir, f"test_2d_viz_sample_{i}.png")
        visualize_sample_2d(depth, gt_traj, pred_traj, pred_uvd, save_path, sample_id=i)

    print("\n完成。")


if __name__ == "__main__":
    main()
