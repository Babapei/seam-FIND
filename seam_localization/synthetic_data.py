"""
合成数据：生成带焊缝凹槽的深度图（及可选彩图），用于在仿真相机未就绪时跑通寻位流程。
模拟：平面 + 一条沿行方向的凹槽（焊缝）。
"""
import numpy as np
from typing import Tuple, Optional

from .camera_utils import CameraIntrinsics


def make_synthetic_depth_with_seam(
    height: int = 240,
    width: int = 320,
    base_depth: float = 1.0,
    seam_depth_delta: float = -0.01,
    seam_width_px: float = 3.0,
    seam_center_col: Optional[float] = None,
    noise_std: float = 0.0,
    invalid_ratio: float = 0.0,
) -> np.ndarray:
    """
    生成带一条“焊缝”凹槽的深度图（单位：米）。

    Args:
        height, width: 图像尺寸
        base_depth: 平面基础深度（米）
        seam_depth_delta: 缝相对平面的深度差（负=凹槽），米
        seam_width_px: 缝在图像上的宽度（像素），用高斯模拟
        seam_center_col: 缝中心列坐标，None 则取 width/2
        noise_std: 高斯噪声标准差（米），0 表示无噪声
        invalid_ratio: 随机无效像素比例 [0,1)

    Returns:
        depth: (H, W)，单位米
    """
    if seam_center_col is None:
        seam_center_col = width / 2.0
    u = np.arange(width, dtype=np.float64)
    v = np.arange(height, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)
    # 凹槽：沿每行在 seam_center_col 附近深度最小
    dist_sq = (uu - seam_center_col) ** 2
    groove = seam_depth_delta * np.exp(-dist_sq / (2 * (seam_width_px ** 2)))
    depth = base_depth + groove
    if noise_std > 0:
        depth = depth + np.random.randn(height, width).astype(np.float64) * noise_std
    if invalid_ratio > 0:
        mask = np.random.rand(height, width) < invalid_ratio
        depth[mask] = 0.0  # 无效用 0
    return depth.astype(np.float64)


def make_synthetic_rgb_with_seam(
    height: int,
    width: int,
    seam_center_col: Optional[float] = None,
    background_rgb: Tuple[int, int, int] = (80, 80, 80),
    seam_rgb: Tuple[int, int, int] = (40, 40, 50),
    seam_width_px: float = 4.0,
) -> np.ndarray:
    """
    生成与深度图对齐的彩图：缝处颜色略深（模拟焊缝纹理）。
    """
    if seam_center_col is None:
        seam_center_col = width / 2.0
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = background_rgb[0]
    rgb[:, :, 1] = background_rgb[1]
    rgb[:, :, 2] = background_rgb[2]
    u = np.arange(width, dtype=np.float64)
    dist = np.abs(u - seam_center_col)
    blend = np.exp(-(dist ** 2) / (2 * seam_width_px ** 2))
    blend_2d = np.broadcast_to(blend, (height, width))
    for c in range(3):
        rgb[:, :, c] = np.clip(
            (1 - blend_2d) * background_rgb[c] + blend_2d * seam_rgb[c],
            0,
            255,
        ).astype(np.uint8)
    return rgb


def default_intrinsics(width: int = 320, height: int = 240) -> CameraIntrinsics:
    """默认虚拟相机内参（与合成图尺寸一致）。"""
    fx = fy = width * 1.2
    cx = width / 2.0
    cy = height / 2.0
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)
