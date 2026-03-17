"""depth_valley：深度图凹槽提取，调用 seam_localization"""
import os
import sys
import numpy as np

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

from seam_localization.seam_from_depth import extract_seam_from_depth


def extract_seam_depth_valley(depth, direction="row", kernel_size=5, **kwargs):
    """返回 (M,3) u,v,depth"""
    return extract_seam_from_depth(
        depth, direction=direction, kernel_size=kernel_size, **kwargs
    )
