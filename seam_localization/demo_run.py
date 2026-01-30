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

    # 2. 寻位：深度图 + 内参 -> 3D 轨迹
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

    # 3. 可选：保存轨迹供后续工艺使用
    out_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(out_dir, exist_ok=True)
    traj_path = os.path.join(out_dir, "seam_trajectory_cam.txt")
    np.savetxt(traj_path, trajectory_3d, fmt="%.6f", header="x y z (camera frame, meters)")
    print("3. 轨迹已保存:", traj_path)
    print("\n完成。后续可接入：仿真相机的深度/点云、标定后的真实内参、或深度学习焊缝分割。")


if __name__ == "__main__":
    main()
