"""laser_line：split-and-merge 风格 2D 缝线提取，参考 laser_line_extraction"""
import numpy as np
from typing import Tuple


def extract_seam_laser_line(
    depth: np.ndarray,
    direction: str = "row",
    min_line_points: int = 9,
    max_line_gap: float = 0.4,
    min_split_dist: float = 0.05,
    **kwargs,
) -> np.ndarray:
    """
    对深度图每行/列做 split-and-merge，取最接近中心的线段中点作为缝点。
    返回 (M,3) u,v,depth
    """
    H, W = depth.shape
    seam_uvd = []
    axis = 0 if direction == "row" else 1
    n_lines = H if direction == "row" else W
    for i in range(n_lines):
        if direction == "row":
            profile = depth[i, :]
        else:
            profile = depth[:, i]
        valid = np.isfinite(profile) & (profile > 0)
        idx = np.where(valid)[0]
        if len(idx) < min_line_points:
            continue
        vals = profile[idx]
        u_or_v = idx.astype(np.float64)
        center = (u_or_v[0] + u_or_v[-1]) / 2
        segments = _split_merge(u_or_v, vals, min_split_dist, max_line_gap)
        best_mid = None
        best_dist = np.inf
        for seg_u, seg_d in segments:
            if len(seg_u) < min_line_points:
                continue
            mid_u = np.median(seg_u)
            mid_d = np.median(seg_d)
            d = abs(mid_u - center)
            if d < best_dist:
                best_dist = d
                best_mid = (mid_u, mid_d)
        if best_mid is not None:
            uu, dd = best_mid
            if direction == "row":
                seam_uvd.append([uu, float(i), dd])
            else:
                seam_uvd.append([float(i), uu, dd])
    return np.array(seam_uvd, dtype=np.float64) if seam_uvd else np.zeros((0, 3))


def _split_merge(u: np.ndarray, d: np.ndarray, min_split: float, max_gap: float) -> list:
    """简单 split-and-merge，返回 [(u_seg, d_seg), ...]"""
    segments = []
    i = 0
    while i < len(u):
        j = i
        while j + 1 < len(u) and u[j + 1] - u[j] <= max_gap:
            j += 1
        seg_u = u[i : j + 1]
        seg_d = d[i : j + 1]
        if len(seg_u) >= 2:
            segments.append((seg_u, seg_d))
        i = j + 1
    return segments
