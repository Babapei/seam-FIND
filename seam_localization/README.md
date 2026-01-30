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
├── camera_utils.py      # 相机内参、深度→点云、像素反投影
├── seam_from_depth.py   # 2D 缝线提取：深度图凹槽/边缘、图像边缘
├── pipeline.py          # 端到端：depth(+rgb)+内参 → 3D 轨迹
├── synthetic_data.py    # 合成带焊缝的深度图/彩图（仿真相机未就绪时用）
├── demo_run.py          # 最小可跑 Demo
├── README.md
└── output/              # Demo 输出轨迹 (seam_trajectory_cam.txt)
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

## 与推荐仓库的关系

| 仓库 | 用途 | 本模块中的位置 |
|------|------|----------------|
| **laser_line_extraction** | 2D 线段提取（split-and-merge） | 思路一致：2D 找线；本模块用「深度图凹槽/边缘」做缝线，可后续对接 ROS 或移植其算法 |
| **PointNet / PointNet++ / DGCNN** | 点云分类/分割 | 后续 3D 流派：点云焊缝分割或线提取后可接入本模块的相机反投影，或直接输出轨迹 |

当前先把 **2D 流派 + 合成数据** 跑通；仿真相机与标定就绪后，只需把 `depth/rgb/intrinsics` 换成仿真输出即可。
