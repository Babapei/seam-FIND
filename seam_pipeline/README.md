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

---

## 各方法所需数据格式

不同方法对输入数据的要求不同，用户需按所选方法准备对应格式。

### pointnet / pointnet2 / dgcnn（点云深度学习）

| 阶段 | 数据格式 | 说明 |
|------|----------|------|
| **训练** | `.npy` 文件，内容为 dict | `{"points": (N_frame, P, 3), "labels": (N_frame, P)}`。`points` 为点云，相机系 xyz，单位米；`labels` 为 0/1，0=背景、1=缝；`P` 通常 2048 |
| **推理** | `points` 数组 `(N, 3)` | 单帧点云，相机系 xyz，单位米。可从深度图反投影得到，或直接提供 `.npy` |

**数据来源**：深度图 + 相机内参 → `depth_to_point_cloud` 反投影 → 下采样到 P 点。缝在合成数据中为平面凹槽，反投影后 x≈0 的点标为缝。

---

### depth_valley（深度凹槽）

| 项目 | 格式 | 说明 |
|------|------|------|
| **输入** | 深度图 `(H, W)`，`float` | 单位米。缝为工件表面的**凹槽**，在深度图中表现为每行的**最小值**（深度更小） |
| **配合** | 相机内参 | `fx, fy, cx, cy`，用于 uvd 反投影到 3D |
| **数据来源** | 深度相机 | 直接输出的深度图。缝必须是平面上的凹槽（如 V 型坡口、焊缝凹陷） |

**典型场景**：纯深度相机，无激光。缝通过深度凹陷体现。

---

### laser_line（激光条纹 / split-and-merge）

| 项目 | 格式 | 说明 |
|------|------|------|
| **输入** | 深度图 `(H, W)`，`float` | 单位米。设计上适合**激光条纹**：只有沿缝的一条线有有效深度，其余为 0 或无效 |
| **配合** | 相机内参 | 用于 uvd 反投影到 3D |
| **参数** | `max_line_gap` | 相邻有效像素索引差 ≤ 此值才会合并。dense 深度图需 ≥ 1（如 2.0） |
| **数据来源** | 激光 + 相机 | 激光线打在工件表面，相机只在该线处有有效深度 |

**典型场景**：主动激光线扫描。缝处激光线发生形变，提取该线即得缝轨迹。若使用 dense 深度图（全图有效），需在 config 中设置 `max_line_gap: 2.0` 或更大。

---

### 总结对比

| 方法 | 输入类型 | 缝的物理形态 | 典型传感器 |
|------|----------|--------------|------------|
| pointnet / pointnet2 / dgcnn | 点云 (N,3) | 凹槽（点云中 x≈0） | 深度相机 → 反投影 |
| depth_valley | 深度图 (H,W) | 凹槽（每行深度最小） | 深度相机 |
| laser_line | 深度图 (H,W) | 激光线形变（有效深度沿一条线） | 激光线 + 深度相机 |

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
├── output/               # 现有实验结果
├── train.py              # 统一训练入口
├── inference.py          # 统一推理入口
├── test_with_viz.py      # 深度学习模型测试验证可视化(训练完成后)
├── test_2d_with_viz.py   # 2d模型测试验证可视化(直接测试)
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



## 注意

### 深度学习模型

| 模型 | 与原仓库的差异 |
|------|----------------|
| **PointNet** | 结构基本按 pointnet-master sem_seg，只是把类别从 13 改成 2，**改动较小**。 |
| **PointNet++** | 有明显简化：1）用 **KNN 代替 ball query**（原版是半径查询），邻域定义不同；2）`farthest_point_sample` 用 Python 循环实现，原版是 CUDA。算法逻辑在，但实现上是近似。 |
| **DGCNN** | 原 dgcnn-master 的 PyTorch 版做的是**分类**，其 TF sem_seg 有另一套结构。当前实现是把分类改成逐点分割，按思路加了分割头，**并非原版 sem_seg 的完整复现**。 |

### 2D 方法

| 方法 | 与原仓库的差异 |
|------|----------------|
| **laser_line** | 相比 laser_line_extraction 有较大简化：原版有 outlier 过滤、完整 split-and-merge、Pfister 加权线拟合、迭代最小二乘、协方差等，目前实现只做了基于 `max_line_gap` 的简单分割，没有线拟合和迭代优化。 |

---

