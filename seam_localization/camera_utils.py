"""
相机内参与深度图反投影工具。
用于：深度图 + 相机内参 -> 点云；像素 (u,v) + 深度 -> 3D 点。
与 3D 视觉传感器仿真输出的接口一致（后续可替换为仿真器的标定结果）。
"""
import numpy as np
from typing import Optional, Tuple


class CameraIntrinsics:
    """相机内参 (fx, fy, cx, cy)，深度单位为米。"""

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.width = width
        self.height = height

    def to_matrix(self) -> np.ndarray:
        """3x3 内参矩阵 K."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ], dtype=np.float64)

    @classmethod
    def from_matrix(cls, K: np.ndarray) -> "CameraIntrinsics":
        """从 3x3 矩阵构造."""
        return cls(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
        )


def unproject_pixel(
    u: float,
    v: float,
    depth: float,
    intrinsics: CameraIntrinsics,
) -> Tuple[float, float, float]:
    """
    单像素反投影到相机坐标系 3D 点 (x, y, z)。
    假设深度 depth 为沿 z 轴的正值（单位与标定一致，通常为米）。
    """
    if depth <= 0 or not np.isfinite(depth):
        return (np.nan, np.nan, np.nan)
    x = (u - intrinsics.cx) * depth / intrinsics.fx
    y = (v - intrinsics.cy) * depth / intrinsics.fy
    z = depth
    return (x, y, z)


def depth_to_point_cloud(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    rgb: Optional[np.ndarray] = None,
    depth_scale: float = 1.0,
    depth_invalid: float = 0.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    深度图反投影为相机坐标系下的点云。

    Args:
        depth: (H, W)，深度图，单位与 depth_scale 配合（如毫米则 depth_scale=0.001）
        intrinsics: 相机内参
        rgb: 可选 (H, W, 3)，与点云一一对应的颜色
        depth_scale: 深度图数值 * depth_scale = 米（若 depth 已是米则为 1.0）
        depth_invalid: 视为无效的深度值（不参与生成点）

    Returns:
        points: (N, 3) 相机坐标系 x,y,z
        colors: (N, 3) 或 None
    """
    h, w = depth.shape
    u = np.arange(w, dtype=np.float64)
    v = np.arange(h, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)
    z = depth.astype(np.float64) * depth_scale
    valid = (z > 0) & np.isfinite(z) & (z != depth_invalid * depth_scale)
    uu = uu[valid]
    vv = vv[valid]
    zz = z[valid]
    x = (uu - intrinsics.cx) * zz / intrinsics.fx
    y = (vv - intrinsics.cy) * zz / intrinsics.fy
    points = np.stack([x, y, zz], axis=1)
    colors = None
    if rgb is not None and rgb.shape[:2] == (h, w):
        if rgb.ndim == 3:
            colors = rgb[valid]
        else:
            colors = rgb[valid]
    return points, colors
