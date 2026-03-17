# seam-FIND / 焊缝寻位

焊缝寻位是焊接机器人视觉导引的核心环节：通过深度相机或结构光采集工件表面数据（深度图或点云），识别焊缝位置并输出 3D 轨迹，供焊枪位姿规划和焊缝跟踪使用。典型应用包括 V 型坡口、对接焊缝、角焊缝等，需要将缝线从 2D 图像或 3D 点云中稳定提取并反投影到世界坐标系。

本仓库面向**焊缝 3D 轨迹提取**这一目标，实现了一套**工程化流程pipeline**：统一配置、多模型可选、一条线集成——用户提供数据 → 选择模型训练 → 保存权重 → 推理输出 3D 轨迹。同时支持两种技术路线：一是从深度图出发的 2D 传统方法（基于几何特征，无需训练）；二是直接从点云出发的深度学习分割（PointNet 系，需少量标注数据）。在 3D 视觉仿真尚未就绪时，使用合成数据验证各方法的可行性与精度。

## 项目目标

在 3D 视觉仿真尚未就绪时，用**合成数据**验证多种焊缝提取路线，为后续接入真实传感器和仿真相机做准备。本仓库统一实现：

- **2D 传统方法**：从深度图直接提取缝线（凹槽取最小深度、激光条纹 split-and-merge），反投影到 3D 得到轨迹。
- **点云深度学习**：PointNet / PointNet++ / DGCNN 对点云做缝/背景二类分割，缝点组成 3D 轨迹。

数据流：**深度图或点云** → **缝线/缝点提取** → **3D 轨迹** → 焊枪位姿。

## 支持的方法

| 类型 | 方法 | 输入 | 需训练 |
|------|------|------|--------|
| 2D 传统 | depth_valley | 深度图 | 否 |
| 2D 传统 | laser_line | 深度图 | 否 |
| 点云深度学习 | pointnet / pointnet2 / dgcnn | 点云 (N,3) | 是 |

各方法的数据格式、训练与推理说明见 [seam_pipeline/README.md](seam_pipeline/README.md)。

## 目录

| 目录 | 说明 |
|------|------|
| **seam_pipeline/** | 主流程：配置、训练、推理、测试。详见 [seam_pipeline/README.md](seam_pipeline/README.md) |
| **seam_localization/** | 底层模块：2D 缝提取、点云/合成数据、相机工具等。详见 [seam_localization/README.md](seam_localization/README.md) |

## 项目结构

```
seam-FIND/
├── README.md
├── requirements-seam.txt
├── method_compare.md              # 方法对比
├── env_install/                   # 环境配置说明
│   └── env_install.md
├── seam_pipeline/                 # 主流程（工程化 pipeline）
│   ├── config/                    # 模型配置
│   │   ├── default.yaml
│   │   ├── depth_valley.yaml
│   │   └── laser_line.yaml
│   ├── models/                    # PointNet / PointNet++ / DGCNN
│   ├── data/                      # 数据加载
│   ├── extractors/                # 2D 提取器 (depth_valley, laser_line)
│   ├── checkpoints/               # 训练权重
│   ├── output/                    # 测试结果与可视化
│   ├── train.py                   # 训练入口
│   ├── inference.py               # 推理入口
│   ├── test_with_viz.py           # 深度学习测试 + 可视化
│   ├── test_2d_with_viz.py        # 2D 方法测试 + 可视化
│   └── README.md
└── seam_localization/             # pipeline 依赖的底层模块
    ├── camera_utils.py            # 相机内参、深度反投影
    ├── synthetic_data.py          # 合成深度图 / 点云
    ├── seam_from_depth.py         # 2D 缝线提取（depth_valley 调用）
    ├── pointcloud_dataset.py      # 点云数据生成
    └── pointcloud_data/           # 训练数据 seam_train.npy, seam_val.npy
```

方法对比见 [method_compare.md](method_compare.md)。

## 快速开始

具体配置方法见**env_install/**
```bash
pip install -r requirements-seam.txt
python seam_pipeline/inference.py --config seam_pipeline/config/default.yaml --points data/points.npy --out output/traj.txt
```

训练、2D 方法、测试脚本等见 `seam_pipeline/README.md`。

实验结果见**seam_pipeline\output**

---

## eg
![替代文字](seam_pipeline\output\pointnet_test\test_viz_sample_0.png)
![替代文字](seam_pipeline\output\dv_test\test_2d_viz_sample_0.png)
![替代文字](seam_pipeline\output\ll_test\test_2d_viz_sample_1.png)

---

## 参考仓库

https://github.com/charlesq34/pointnet

https://github.com/charlesq34/pointnet2

https://github.com/WangYueFt/dgcnn

https://github.com/kam3k/laser_line_extraction