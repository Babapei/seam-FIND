#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
seam_pipeline 测试脚本：生成测试用例 → 推理 → 输出指标 + 可视化
独立于 train/inference，用于验证训练效果。不修改其他脚本。
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


def load_model_and_predict(model_name, ckpt_path, points, labels, device="cpu", num_point=2048, subsample_seed=42):
    """
    加载模型并对点云推理，返回 (pred, trajectory, pts_used, labels_used)。
    pred/labels_used 与 pts_used 一一对应，用于计算指标。
    """
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
    lab = np.asarray(labels, dtype=np.int64).ravel()
    if lab.shape[0] != pts.shape[0]:
        lab = np.zeros(pts.shape[0], dtype=np.int64)

    rng = np.random.RandomState(subsample_seed)
    if pts.shape[0] > num_point_ckpt:
        idx = rng.choice(pts.shape[0], num_point_ckpt, replace=False)
        pts = pts[idx]
        lab = lab[idx]
    elif pts.shape[0] < num_point_ckpt:
        idx = rng.choice(pts.shape[0], num_point_ckpt, replace=True)
        pts = pts[idx]
        lab = lab[idx]

    with torch.no_grad():
        x = torch.from_numpy(pts).unsqueeze(0).to(device)
        logits = model(x)
        pred = logits[0].argmax(1).cpu().numpy()

    seam_mask = pred == 1
    seam_pts = pts[seam_mask]
    order = np.argsort(seam_pts[:, 1])
    trajectory = seam_pts[order]
    return pred, trajectory, pts, lab


def compute_metrics(pred, labels, num_class=2):
    """Accuracy、IoU (缝、背景)、mean IoU"""
    pred = np.asarray(pred).ravel()
    labels = np.asarray(labels).ravel()
    acc = (pred == labels).mean()

    ious = []
    for c in range(num_class):
        mask_pred = pred == c
        mask_gt = labels == c
        inter = (mask_pred & mask_gt).sum()
        union = (mask_pred | mask_gt).sum()
        iou = inter / union if union > 0 else float("nan")
        ious.append(iou)
    mean_iou = np.nanmean(ious) if any(not np.isnan(i) for i in ious) else float("nan")
    return {"accuracy": acc, "iou_seam": ious[1], "iou_bg": ious[0], "mean_iou": mean_iou}


def visualize_sample(pts, labels, pred, trajectory, save_path, sample_id=0):
    """单样本可视化：点云(gt着色)、点云(预测着色)、3D轨迹"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        from mpl_toolkits.mplot3d import Axes3D

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
    ax1 = fig.add_subplot(131, projection="3d")
    ax2 = fig.add_subplot(132, projection="3d")
    ax3 = fig.add_subplot(133, projection="3d")

    # 下采样以便绘图
    n_show = min(3000, pts.shape[0])
    idx = np.random.RandomState(sample_id).choice(pts.shape[0], n_show, replace=False)

    # 统一 XYZ 范围，用于等比例显示
    all_pts = np.vstack([pts, trajectory]) if trajectory.shape[0] > 0 else pts
    x_min, x_max = all_pts[:, 0].min(), all_pts[:, 0].max()
    y_min, y_max = all_pts[:, 1].min(), all_pts[:, 1].max()
    z_min, z_max = all_pts[:, 2].min(), all_pts[:, 2].max()
    # 避免零范围
    mx = max(x_max - x_min, y_max - y_min, z_max - z_min, 1e-6)
    cx, cy, cz = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
    r = mx / 2
    x_min, x_max = cx - r, cx + r
    y_min, y_max = cy - r, cy + r
    z_min, z_max = cz - r, cz + r

    def set_equal_aspect(ax):
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

    # 左：GT 着色
    pt = pts[idx]
    lb = labels[idx]
    ax1.scatter(pt[lb == 0, 0], pt[lb == 0, 1], pt[lb == 0, 2], c="lightgray", s=2, alpha=0.5)
    ax1.scatter(pt[lb == 1, 0], pt[lb == 1, 1], pt[lb == 1, 2], c="blue", s=5, alpha=0.8)
    ax1.set_title("Ground Truth (蓝=缝)")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
    set_equal_aspect(ax1)

    # 中：预测着色
    pr = pred[idx]
    ax2.scatter(pt[pr == 0, 0], pt[pr == 0, 1], pt[pr == 0, 2], c="lightgray", s=2, alpha=0.5)
    ax2.scatter(pt[pr == 1, 0], pt[pr == 1, 1], pt[pr == 1, 2], c="red", s=5, alpha=0.8)
    ax2.set_title("Prediction (红=缝)")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_zlabel("z")
    set_equal_aspect(ax2)

    # 右：3D 轨迹
    if trajectory.shape[0] > 0:
        ax3.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], "b-", linewidth=1.5)
        ax3.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], c="green", s=40, label="起点")
        ax3.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], c="red", s=40, label="终点")
    ax3.set_title("3D 轨迹")
    ax3.set_xlabel("x"); ax3.set_ylabel("y"); ax3.set_zlabel("z")
    ax3.legend(loc="upper right", fontsize=8)
    set_equal_aspect(ax3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print("  可视化已保存:", save_path)


def main():
    parser = argparse.ArgumentParser(description="seam_pipeline 测试：生成用例→推理→指标+可视化")
    parser.add_argument("--config", type=str, default="seam_pipeline/config/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num_test", type=int, default=20, help="测试用例数量")
    parser.add_argument("--num_viz", type=int, default=3, help="可视化样本数量")
    parser.add_argument("--out_dir", type=str, default="seam_pipeline/output/test")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=6000, help="测试集随机种子起始")
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path) and not os.path.isfile(config_path):
        config_path = os.path.join(_root, config_path)
    cfg = load_config(config_path)
    model_name = cfg["model"]

    if model_name not in ("pointnet", "pointnet2", "dgcnn"):
        print("test_with_viz 仅支持点云模型 (pointnet/pointnet2/dgcnn)，2D 方法请用 inference.py")
        sys.exit(0)

    ckpt = args.checkpoint
    if not ckpt:
        out_cfg = cfg.get("output", {})
        ckpt_dir = out_cfg.get("checkpoint_dir", "seam_pipeline/checkpoints")
        ckpt = os.path.join(_root, ckpt_dir, out_cfg.get("exp_name", "seam_exp"), f"{model_name}_best.pt")
    if not os.path.isfile(ckpt):
        print("Checkpoint 不存在:", ckpt, "\n请先训练: python seam_pipeline/train.py --config ...")
        sys.exit(1)

    num_points = cfg.get("data", {}).get("num_points", 2048)
    from seam_localization.pointcloud_dataset import generate_one_scene

    print("=== seam_pipeline 测试 ===\n模型:", model_name, "| 测试数:", args.num_test)

    all_metrics = []
    viz_samples = []  # (pts, labels, pred, trajectory, i)

    for i in range(args.num_test):
        points, labels = generate_one_scene(num_points_target=num_points, seed=args.seed + i)
        pred, trajectory, pts_used, labels_used = load_model_and_predict(
            model_name, ckpt, points, labels, args.device, num_points
        )
        m = compute_metrics(pred, labels_used)
        all_metrics.append(m)
        if i < args.num_viz:
            viz_samples.append((pts_used, labels_used, pred, trajectory, i))

    # 汇总指标
    acc_mean = np.mean([x["accuracy"] for x in all_metrics])
    iou_seam_mean = np.nanmean([x["iou_seam"] for x in all_metrics])
    iou_bg_mean = np.nanmean([x["iou_bg"] for x in all_metrics])
    mean_iou_avg = np.nanmean([x["mean_iou"] for x in all_metrics])

    out_dir = os.path.join(_root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    metrics_path = os.path.join(out_dir, "test_metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"model: {model_name}\n")
        f.write(f"num_test: {args.num_test}\n")
        f.write(f"accuracy: {acc_mean:.4f}\n")
        f.write(f"iou_seam: {iou_seam_mean:.4f}\n")
        f.write(f"iou_bg: {iou_bg_mean:.4f}\n")
        f.write(f"mean_iou: {mean_iou_avg:.4f}\n")
    print("\n--- 指标 ---")
    print("accuracy:  {:.4f}".format(acc_mean))
    print("iou_seam:  {:.4f}".format(iou_seam_mean))
    print("iou_bg:    {:.4f}".format(iou_bg_mean))
    print("mean_iou:  {:.4f}".format(mean_iou_avg))
    print("指标已保存:", metrics_path)

    for pts, labels, pred, trajectory, i in viz_samples:
        save_path = os.path.join(out_dir, f"test_viz_sample_{i}.png")
        visualize_sample(pts, labels, pred, trajectory, save_path, sample_id=i)

    print("\n完成。")


if __name__ == "__main__":
    main()
