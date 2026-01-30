#!/usr/bin/env python3
"""
焊缝寻位最小可跑 Demo：用合成深度图跑通 2D 流派流程，输出焊缝 3D 轨迹。
不依赖仿真相机，可直接运行验证算法链路。
"""
import numpy as np
import sys
import os

# 允许从项目根目录或 seam_localization 目录运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from seam_localization.synthetic_data import (
    make_synthetic_depth_with_seam,
    make_synthetic_rgb_with_seam,
    default_intrinsics,
)
from seam_localization.pipeline import run_seam_localization
from seam_localization.seam_from_depth import extract_seam_from_depth


def visualize(depth, rgb, trajectory_3d, seam_uvd, out_dir):
    """绘制：合成深度图、2D 缝线叠加、3D 轨迹，并保存到 output。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        # 显式加载 SimHei，避免 matplotlib 找不到系统字体
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for _path in [
            os.path.join(_root, "SimHei.ttf"),
            "/usr/share/fonts/truetype/SimHei.ttf",
        ]:
            if os.path.isfile(_path):
                fm.fontManager.addfont(_path)
                plt.rcParams["font.family"] = fm.FontProperties(fname=_path).get_name()
                break
        plt.rcParams["axes.unicode_minus"] = False
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        print("未安装 matplotlib，跳过可视化。可运行: pip install matplotlib")
        return None

    fig = plt.figure(figsize=(14, 5))

    # 1. 深度图
    ax1 = fig.add_subplot(131)
    im1 = ax1.imshow(depth, cmap="viridis")
    ax1.set_title("合成深度图 (带焊缝凹槽)")
    ax1.set_xlabel("u (列)")
    ax1.set_ylabel("v (行)")
    plt.colorbar(im1, ax=ax1, label="深度 (m)")

    # 2. 深度图 + 2D 缝线叠加
    ax2 = fig.add_subplot(132)
    ax2.imshow(depth, cmap="gray")
    if seam_uvd.shape[0] > 0:
        u, v = seam_uvd[:, 0], seam_uvd[:, 1]
        ax2.plot(u, v, "r-", linewidth=1, alpha=0.9, label="提取的缝线")
        ax2.scatter(u[:: max(1, len(u) // 30)], v[:: max(1, len(v) // 30)], c="red", s=8)
    ax2.set_title("深度图 + 提取的 2D 缝线")
    ax2.set_xlabel("u (列)")
    ax2.set_ylabel("v (行)")
    ax2.legend(loc="upper right", fontsize=8)

    # 3. 3D 轨迹（相机坐标系）
    ax3 = fig.add_subplot(133, projection="3d")
    if trajectory_3d.shape[0] > 0:
        x, y, z = trajectory_3d[:, 0], trajectory_3d[:, 1], trajectory_3d[:, 2]
        ax3.plot(x, y, z, "b-", linewidth=1.5, alpha=0.8)
        ax3.scatter(x[0], y[0], z[0], c="green", s=40, label="起点")
        ax3.scatter(x[-1], y[-1], z[-1], c="red", s=40, label="终点")
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")
    ax3.set_zlabel("z (m)")
    ax3.set_title("焊缝 3D 轨迹 (相机系)")
    ax3.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "seam_demo_visualization.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    print("=== 焊缝寻位 Demo（2D 流派：深度图 -> 缝线 -> 3D 轨迹）===\n")

    # 1. 合成数据：带焊缝凹槽的深度图
    height, width = 240, 320
    depth = make_synthetic_depth_with_seam(
        height=height,
        width=width,
        base_depth=1.0,
        seam_depth_delta=-0.01,
        seam_width_px=4.0,
        seam_center_col=width / 2.0,
        noise_std=0.001,
        invalid_ratio=0.0,
    )
    rgb = make_synthetic_rgb_with_seam(height=height, width=width)
    intrinsics = default_intrinsics(width=width, height=height)

    print("1. 合成深度图尺寸:", depth.shape, "单位: 米")
    print("   焊缝凹槽中心列:", width / 2.0, "深度差约 -0.01 m\n")

    # 2. 寻位：深度图 + 内参 -> 3D 轨迹（并取 2D 缝线用于可视化）
    seam_uvd = extract_seam_from_depth(
        depth,
        direction="row",
        kernel_size=5,
        min_valid_depth=0.1,
        max_valid_depth=10.0,
    )
    trajectory_3d = run_seam_localization(
        depth,
        intrinsics,
        rgb=rgb,
        method="depth_valley",
        direction="row",
        depth_scale=1.0,
        min_valid_depth=0.1,
        max_valid_depth=10.0,
        kernel_size=5,
    )

    print("2. 焊缝 3D 轨迹点数:", len(trajectory_3d))
    if len(trajectory_3d) > 0:
        print("   轨迹范围 (相机坐标系, 米):")
        print("   x: [{:.4f}, {:.4f}]".format(trajectory_3d[:, 0].min(), trajectory_3d[:, 0].max()))
        print("   y: [{:.4f}, {:.4f}]".format(trajectory_3d[:, 1].min(), trajectory_3d[:, 1].max()))
        print("   z: [{:.4f}, {:.4f}]".format(trajectory_3d[:, 2].min(), trajectory_3d[:, 2].max()))
        print("   前 3 点 (x,y,z):")
        for i in range(min(3, len(trajectory_3d))):
            print("     ", trajectory_3d[i].tolist())
    print()

    # 3. 保存轨迹与可视化
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    traj_path = os.path.join(out_dir, "seam_trajectory_cam.txt")
    np.savetxt(traj_path, trajectory_3d, fmt="%.6f", header="x y z (camera frame, meters)")
    print("3. 轨迹已保存:", traj_path)
    viz_path = visualize(depth, rgb, trajectory_3d, seam_uvd, out_dir)
    if viz_path:
        print("   可视化已保存:", viz_path)
    print("\n完成。后续可接入：仿真相机的深度/点云、标定后的真实内参、或深度学习焊缝分割。")


if __name__ == "__main__":
    main()
