# Seam Pipeline — 焊缝寻位工程化流程

统一配置、多模型可选、一条线集成：用户提供数据 → 选择模型训练 → 保存权重 → 推理输出 3D 轨迹。

## 模型选项

| 模型 | 类型 | 输入 | 需训练 |
|------|------|------|--------|
| **pointnet** | 点云深度学习 | 点云 (N,3) | 是 |
| **pointnet2** | 点云深度学习 | 点云 (N,3) | 是 |
| **dgcnn** | 点云深度学习 | 点云 (N,3) | 是 |
| **depth_valley** | 2D 传统 | 深度图 | 否 |
| **laser_line** | 2D 传统 (split-and-merge) | 深度图 | 否 |

所有深度学习模型均用 **PyTorch** 实现，参考 pointnet-master、pointnet2-master、dgcnn-master 结构，不简化。

## 快速开始

### 1. 准备数据

点云格式：`points (N_frame, P, 3)`, `labels (N_frame, P)` 存为 `.npy`，格式为 `{"points": ..., "labels": ...}`。

可用合成数据生成：

```bash
python seam_localization/pointcloud_dataset.py --num_train 200 --num_val 50 --num_points 2048
```

### 2. 配置 config

编辑 `seam_pipeline/config/default.yaml`：

```yaml
model: pointnet   # 或 pointnet2, dgcnn, depth_valley, laser_line
data:
  train_npy: "seam_localization/pointcloud_data/seam_train.npy"
  val_npy: "seam_localization/pointcloud_data/seam_val.npy"
  num_points: 2048
```

### 3. 训练（深度学习模型）

```bash
python seam_pipeline/train.py --config seam_pipeline/config/default.yaml
```

权重保存到 `seam_pipeline/checkpoints/{exp_name}/{model}_best.pt`。

### 4. 推理

```bash
# 点云模型
python seam_pipeline/inference.py --config seam_pipeline/config/default.yaml --points data/points.npy --out output/traj.txt

# 2D 方法（无需 checkpoint）
# 修改 config 中 model 为 depth_valley 或 laser_line 后运行
python seam_pipeline/inference.py --config seam_pipeline/config/default.yaml --depth depth.npy --out output/traj.txt
```

## 目录结构

```
seam_pipeline/
├── config/default.yaml   # 默认配置
├── models/               # PointNet, PointNet++, DGCNN (PyTorch)
├── data/                 # 数据加载
├── extractors/           # 2D 提取器 (depth_valley, laser_line)
├── train.py              # 统一训练入口
├── inference.py          # 统一推理入口
└── README.md
```

## 依赖

- torch, numpy, pyyaml
- seam_localization（用于 2D 方法、相机工具、合成数据）


## 测试

### 深度学习模型 (test_with_viz.py)

```
# 使用默认 config，自动从 checkpoints 读取
python seam_pipeline/test_with_viz.py

# 指定 checkpoint 和测试数量
python seam_pipeline/test_with_viz.py --checkpoint seam_pipeline/checkpoints/seam_exp/pointnet_best.pt --num_test 50 --num_viz 5

# 指定输出目录
python seam_pipeline/test_with_viz.py --out_dir seam_pipeline/output/my_test
```

### 2D 方法 (test_2d_with_viz.py)

专门测试 depth_valley、laser_line，无需 checkpoint，生成合成深度图 → 提取 → 轨迹距离指标 + 可视化。

```
# 指定提取器（覆盖 config 中的 model）
python seam_pipeline/test_2d_with_viz.py --extractor depth_valley

python seam_pipeline/test_2d_with_viz.py --extractor laser_line

# 或使用 2D 专用 config
python seam_pipeline/test_2d_with_viz.py --config seam_pipeline/config/depth_valley.yaml

# 自定义测试数量、输出目录
python seam_pipeline/test_2d_with_viz.py --extractor depth_valley --num_test 50 --out_dir seam_pipeline/output/test_2d
```

输出：`seam_pipeline/output/test_2d/` 下的 `test_2d_metrics.txt` 与 `test_2d_viz_sample_*.png`。
