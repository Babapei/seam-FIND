"""
焊缝寻位端到端流程：深度图（+ 可选彩图） + 相机内参 -> 焊缝 3D 轨迹。
2D 流派：深度图/图像上提取 2D 缝线 -> 反投影到 3D。
"""
import numpy as np
from typing import Optional, Literal

from .camera_utils import CameraIntrinsics, unproject_pixel
from .seam_from_depth import (
    extract_seam_from_depth,
    extract_seam_from_depth_edge,
    extract_seam_from_image,
)


def run_seam_localization(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    rgb: Optional[np.ndarray] = None,
    method: Literal["depth_valley", "depth_edge", "image_edge"] = "depth_valley",
    direction: str = "row",
    depth_scale: float = 1.0,
    min_valid_depth: float = 0.0,
    max_valid_depth: float = 1e6,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    从深度图（及可选彩图）得到焊缝 3D 轨迹（相机坐标系）。

    Args:
        depth: (H, W) 深度图，单位与 depth_scale 一致（如毫米则 depth_scale=0.001）
        intrinsics: 相机内参
        rgb: 可选 (H, W, 3)，method="image_edge" 时使用
        method: "depth_valley" 凹槽局部最小深度；"depth_edge" 深度梯度；"image_edge" 图像边缘
        direction: "row" 每行一个点（缝沿列）；"col" 每列一个点（缝沿行）
        depth_scale: 深度数值 * depth_scale = 米
        min_valid_depth, max_valid_depth: 有效深度范围（原始单位）
        kernel_size: 局部极小/平滑窗口

    Returns:
        trajectory_3d: (N, 3) 相机坐标系下焊缝轨迹点 (x, y, z)，单位米
    """
    if method == "depth_valley":
        seam_uvd = extract_seam_from_depth(
            depth,
            direction=direction,
            kernel_size=kernel_size,
            min_valid_depth=min_valid_depth,
            max_valid_depth=max_valid_depth,
        )
    elif method == "depth_edge":
        seam_uvd = extract_seam_from_depth_edge(
            depth,
            direction=direction,
            blur_size=kernel_size,
            min_valid_depth=min_valid_depth,
            max_valid_depth=max_valid_depth,
        )
    elif method == "image_edge" and rgb is not None:
        seam_uvd = extract_seam_from_image(
            rgb,
            depth,
            use_edges=True,
            min_valid_depth=min_valid_depth,
        )
    else:
        seam_uvd = extract_seam_from_depth(
            depth,
            direction=direction,
            kernel_size=kernel_size,
            min_valid_depth=min_valid_depth,
            max_valid_depth=max_valid_depth,
        )

    if seam_uvd.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)

    trajectory_3d = []
    for i in range(seam_uvd.shape[0]):
        u, v, d = seam_uvd[i, 0], seam_uvd[i, 1], seam_uvd[i, 2] * depth_scale
        x, y, z = unproject_pixel(u, v, d, intrinsics)
        if np.isfinite(x):
            trajectory_3d.append([x, y, z])
    return np.array(trajectory_3d, dtype=np.float64)
