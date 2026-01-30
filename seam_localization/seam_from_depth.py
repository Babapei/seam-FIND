"""
2D 焊缝线提取：在深度图或图像上找到缝线像素 (u, v)，供反投影到 3D。
- 深度图：焊缝多为凹槽，按行/列取局部最小值（谷线）或梯度极大（边缘）。
- 图像：边缘检测、霍夫线等，再在深度图上取深度。
与结构光“光条中心提取”思路一致：先在 2D 找线，再反投影。
"""
import numpy as np
from typing import List, Tuple, Optional

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


def extract_seam_from_depth(
    depth: np.ndarray,
    direction: str = "row",
    kernel_size: int = 5,
    min_valid_depth: float = 0.0,
    max_valid_depth: float = 1e6,
) -> np.ndarray:
    """
    在深度图上按行或按列提取焊缝线（凹槽 = 局部最小深度）。

    Args:
        depth: (H, W) 深度图
        direction: "row" 每行取一个点（缝沿列方向），"col" 每列取一个点（缝沿行方向）
        kernel_size: 局部最小值的邻域大小（奇数）
        min_valid_depth, max_valid_depth: 有效深度范围，之外不参与取最小

    Returns:
        seam_uvd: (N, 3) 每行为 (u, v, depth)，像素坐标 + 深度
    """
    valid = np.isfinite(depth) & (depth >= min_valid_depth) & (depth <= max_valid_depth)
    depth_safe = np.where(valid, depth, np.nan)

    half = kernel_size // 2
    seam_uvd: List[Tuple[float, float, float]] = []

    if direction == "row":
        for v in range(depth.shape[0]):
            row = depth_safe[v, :]
            if not np.any(np.isfinite(row)):
                continue
            # 局部最小值：在窗口内取最小深度的 u
            u_min = -1
            d_min = np.inf
            for u in range(half, depth.shape[1] - half):
                window = row[u - half : u + half + 1]
                if not np.any(np.isfinite(window)):
                    continue
                m = np.nanmin(window)
                if m < d_min:
                    d_min = m
                    u_min = u
            if u_min >= 0 and np.isfinite(d_min):
                seam_uvd.append((float(u_min), float(v), float(d_min)))
    else:  # col
        for u in range(depth.shape[1]):
            col = depth_safe[:, u]
            if not np.any(np.isfinite(col)):
                continue
            v_min = -1
            d_min = np.inf
            for v in range(half, depth.shape[0] - half):
                window = col[v - half : v + half + 1]
                if not np.any(np.isfinite(window)):
                    continue
                m = np.nanmin(window)
                if m < d_min:
                    d_min = m
                    v_min = v
            if v_min >= 0 and np.isfinite(d_min):
                seam_uvd.append((float(u), float(v_min), float(d_min)))

    if not seam_uvd:
        return np.zeros((0, 3), dtype=np.float64)
    return np.array(seam_uvd, dtype=np.float64)


def extract_seam_from_depth_edge(
    depth: np.ndarray,
    direction: str = "row",
    blur_size: int = 5,
    min_valid_depth: float = 0.0,
    max_valid_depth: float = 1e6,
) -> np.ndarray:
    """
    用深度梯度（边缘）找缝：缝两侧深度突变。按行/列找梯度最大的位置。
    适用于“台阶缝”或明显高度差的焊缝。

    Returns:
        seam_uvd: (N, 3) (u, v, depth)
    """
    valid = np.isfinite(depth) & (depth >= min_valid_depth) & (depth <= max_valid_depth)
    depth_safe = np.where(valid, depth.astype(np.float64), np.nan)
    depth_fill = np.nan_to_num(depth_safe, nan=np.nanmean(depth_safe[np.isfinite(depth_safe)]))
    if not HAS_OPENCV:
        # 简单差分
        if direction == "row":
            grad = np.abs(np.diff(depth_fill.astype(np.float32), axis=1))
            grad = np.pad(grad, ((0, 0), (0, 1)), constant_values=0)
        else:
            grad = np.abs(np.diff(depth_fill.astype(np.float32), axis=0))
            grad = np.pad(grad, ((0, 1), (0, 0)), constant_values=0)
    else:
        blurred = cv2.GaussianBlur(depth_fill.astype(np.float32), (blur_size, blur_size), 0)
        if direction == "row":
            grad = np.abs(cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3))
        else:
            grad = np.abs(cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3))

    seam_uvd: List[Tuple[float, float, float]] = []
    if direction == "row":
        for v in range(grad.shape[0]):
            row = grad[v, :]
            u = int(np.argmax(row))
            if np.isfinite(depth[v, u]) and depth[v, u] >= min_valid_depth:
                seam_uvd.append((float(u), float(v), float(depth[v, u])))
    else:
        for u in range(grad.shape[1]):
            col = grad[:, u]
            v = int(np.argmax(col))
            if np.isfinite(depth[v, u]) and depth[v, u] >= min_valid_depth:
                seam_uvd.append((float(u), float(v), float(depth[v, u])))

    if not seam_uvd:
        return np.zeros((0, 3), dtype=np.float64)
    return np.array(seam_uvd, dtype=np.float64)


def extract_seam_from_image(
    image: np.ndarray,
    depth: np.ndarray,
    use_edges: bool = True,
    min_valid_depth: float = 0.0,
) -> np.ndarray:
    """
    在 RGB/灰度图上用边缘或线检测得到候选像素，再在 depth 上取深度。
    适用于缝在纹理/颜色上更明显的场景。

    Args:
        image: (H, W) 或 (H, W, 3)
        depth: (H, W)，与 image 对齐
        use_edges: True 用 Canny 边缘，False 可扩展为霍夫线

    Returns:
        seam_uvd: (N, 3) (u, v, depth)，仅保留深度有效的点
    """
    if not HAS_OPENCV:
        # 无 OpenCV 时退回深度图局部最小
        gray = image if image.ndim == 2 else np.mean(image, axis=2)
        return extract_seam_from_depth(depth, direction="row", min_valid_depth=min_valid_depth)

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.astype(np.uint8)
    edges = cv2.Canny(gray, 50, 150)
    ys, xs = np.where(edges > 0)
    seam_uvd = []
    for u, v in zip(xs, ys):
        d = depth[v, u]
        if np.isfinite(d) and d >= min_valid_depth:
            seam_uvd.append((float(u), float(v), float(d)))
    if not seam_uvd:
        return np.zeros((0, 3), dtype=np.float64)
    # 按 v 排序得到大致沿缝的序列（可选：再按 v 分组取重心成一条线）
    seam_uvd = np.array(seam_uvd)
    order = np.lexsort((seam_uvd[:, 0], seam_uvd[:, 1]))
    return seam_uvd[order]
