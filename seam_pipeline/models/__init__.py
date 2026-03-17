# 焊缝寻位 Pipeline 统一模型注册
# pointnet, pointnet2, dgcnn 均为 PyTorch 实现，焊缝二类分割
from .pointnet_seg import PointNetSeg
from .dgcnn_seg import DGCNNSeg

# PointNet2 可选，若需编译则延迟导入
try:
    from .pointnet2_seg import PointNet2Seg
    HAS_POINTNET2 = True
except ImportError:
    HAS_POINTNET2 = False

_MODELS = {
    "pointnet": PointNetSeg,
    "dgcnn": DGCNNSeg,
}

if HAS_POINTNET2:
    _MODELS["pointnet2"] = PointNet2Seg


def get_model(name, **kwargs):
    if name not in _MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list(_MODELS.keys())}")
    return _MODELS[name](**kwargs)
