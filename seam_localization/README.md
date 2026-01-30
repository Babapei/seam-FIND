# 焊缝寻位 (Seam Localization)

基于仿真的点云/深度图，集成焊缝识别算法，**输出焊缝 3D 轨迹**。  
放在「3D 视觉传感器仿真」之后：仿真得到彩图、深度图、点云后，本模块负责寻缝并得到 3D 轨迹。

## 思路概览

- **2D 流派（当前实现）**：深度图/图像 → 在 2D 上找缝线（凹槽/边缘）→ 像素 (u,v) + 深度 → 反投影到 3D → 轨迹。  
  计算轻、易调试，与结构光「先有图、后有点云」一致，可验证光条/缝线提取。
- **3D 流派（后续扩展）**：直接对点云做线提取（如 PCL、PointNet 等），或对深度图做 CNN 焊缝分割再反投影。

## 目录结构

```
seam_localization/
├── __init__.py
├── camera_utils.py         # 相机内参、深度→点云、像素反投影
├── seam_from_depth.py      # 2D 缝线提取：深度图凹槽/边缘、图像边缘
├── pipeline.py             # 端到端：depth(+rgb)+内参 → 3D 轨迹
├── synthetic_data.py       # 合成带焊缝的深度图/彩图（仿真相机未就绪时用）
├── demo_run.py             # 最小可跑 Demo（2D 流派）
├── pointcloud_dataset.py   # 点云数据生成：深度→点云+二类标签（缝/背景）
├── pointnet_seam_model.py  # PointNet 二类分割模型（PyTorch，基于 pointnet-master）
├── pointnet_seam_data.py   # 点云分割 DataLoader
├── train_pointnet_seam.py  # 训练 PointNet 焊缝分割
├── run_seam_from_pointnet.py # 推理：点云→预测缝点→3D 轨迹
├── pointcloud_data/        # 生成的数据 seam_train.npy, seam_val.npy
├── checkpoints/            # 训练好的 pointnet_seam.pt
├── README.md
└── output/                 # Demo / 推理输出轨迹
```

## 依赖

- **必选**：`numpy`
- **可选**：`opencv-python`（用于 `depth_edge` / `image_edge` 方法及后续扩展）

```bash
pip install numpy opencv-python
```

## 快速跑通（不依赖仿真相机）

在项目根目录执行：

```bash
python seam_localization/demo_run.py
```

使用合成深度图（平面 + 焊缝凹槽）跑通：**深度图 → 2D 缝线（局部最小深度）→ 反投影 → 3D 轨迹**，并保存到 `seam_localization/output/seam_trajectory_cam.txt`。

## 接口说明

### 端到端：`run_seam_localization`

```python
from seam_localization import run_seam_localization
from seam_localization.camera_utils import CameraIntrinsics

# depth: (H, W)，单位与 depth_scale 一致
# intrinsics: 仿真或标定得到的相机内参
trajectory_3d = run_seam_localization(
    depth,
    intrinsics,
    rgb=None,           # 可选，method="image_edge" 时用
    method="depth_valley",  # depth_valley | depth_edge | image_edge
    direction="row",    # 缝沿列方向用 row，沿行方向用 col
    depth_scale=1.0,   # 深度数值 * depth_scale = 米
    min_valid_depth=0.0,
    max_valid_depth=1e6,
    kernel_size=5,
)
# trajectory_3d: (N, 3) 相机坐标系下焊缝轨迹 (x,y,z)，单位米
```

### 与仿真/标定对接

- **输入**：仿真得到的 `depth`、`rgb`（可选）、标定或虚拟的 `CameraIntrinsics`（fx, fy, cx, cy）。
- **输出**：焊缝 3D 轨迹，可再转换到机器人/世界坐标系用于路径规划。

## 点云流派（PointNet 焊缝分割）

在仿真相机/点云尚未就绪时，可**自己生成点云数据**并用 **PointNet** 做二类分割（缝 vs 背景），再得到 3D 轨迹。实现参考了 **pointnet-master** 的 sem_seg/part_seg 结构，用 PyTorch 重写便于跑通。

### 1. 生成点云数据（合成）

从现有合成深度图反投影得到点云，并按「缝在相机系 x≈0」打二类标签：

```bash
python seam_localization/pointcloud_dataset.py --num_train 200 --num_val 50 --num_points 2048
```

会在 `seam_localization/pointcloud_data/` 下生成 `seam_train.npy`、`seam_val.npy`。

### 2. 训练 PointNet 二类分割

```bash
pip install torch
python seam_localization/train_pointnet_seam.py --epochs 30 --batch_size 16
```

模型保存到 `seam_localization/checkpoints/pointnet_seam.pt`。

### 3. 推理：点云 → 缝点 → 3D 轨迹

```bash
python seam_localization/run_seam_from_pointnet.py
```

不指定 `--points` 时用合成一帧点云；也可传入 `.npy` 点云路径。轨迹输出到 `output/seam_trajectory_pointnet.txt`。

### 与推荐仓库的关系

| 仓库 | 用途 | 本模块中的位置 |
|------|------|----------------|
| **laser_line_extraction** | 2D 线段提取（split-and-merge） | 思路一致：2D 找线；本模块用「深度图凹槽/边缘」做缝线，可后续对接 ROS 或移植其算法 |
| **pointnet-master** | PointNet 分类/分割（TF） | 本模块 `pointnet_seam_model.py` 为其 sem_seg 结构的 PyTorch 二类版，数据格式兼容自生成 .npy |
| **pointnet2-master** | PointNet++（需编译 tf_ops） | 后续可替换为 PointNet++ 分割以提升效果 |
| **dgcnn-master** | DGCNN 分类（PyTorch/TF） | 后续可接 DGCNN 做分割或分类 |

当前先把 **2D 流派 + 点云 PointNet 二类分割** 跑通；仿真相机与真实点云就绪后，只需把数据源换成仿真/真机点云即可。
