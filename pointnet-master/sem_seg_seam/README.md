# PointNet 焊缝二类分割（直接用 pointnet-master）

用 **pointnet-master 原版 sem_seg 结构**训练焊缝（缝=1 / 背景=0），只需提供训练集 h5 即可。

## 数据格式

与 `sem_seg` 一致：

- **data**: `(N, num_point, 9)`，float32。前 3 维为 xyz，后 6 维可填 0。
- **label**: `(N, num_point)`，uint8 或 int32，取值 0 或 1。

## 使用步骤

### 1. 在 seam_localization 里生成并导出数据

在项目根目录（seam-FIND）执行：

```bash
# 生成点云 + 二类标签
python seam_localization/pointcloud_dataset.py --num_train 200 --num_val 50 --num_points 2048

# 导出为 pointnet 的 h5 格式，写入 pointnet-master/sem_seg_seam_data/
python seam_localization/export_seam_to_pointnet_h5.py
```

会在 `pointnet-master/sem_seg_seam_data/` 下生成 `seam_train.h5`、`seam_val.h5`。

### 2. 用 pointnet-master 直接训练

```bash
cd pointnet-master/sem_seg_seam
python train_seam.py --num_point 2048 --batch_size 24 --max_epoch 50
```

模型和日志保存在 `log_seam/`，例如 `model.ckpt`。

### 3. 推理 / 接入焊缝轨迹

训练得到的 ckpt 是 **TensorFlow 1.x** 格式。若要在 seam_localization 的 demo 里用，可：

- 继续用当前 **PyTorch 简易版**（`seam_localization/train_pointnet_seam.py`）做推理和可视化；或  
- 在 pointnet-master 下写一个 `batch_inference.py` 读 ckpt、读 h5/点云、输出每点类别，再在外部把缝点连成 3D 轨迹。

## 与 sem_seg 原版的区别

- **NUM_CLASSES = 2**（缝 / 背景），原版为 13（室内语义）。
- **数据来源**：从 `sem_seg_seam_data/seam_train.h5`、`seam_val.h5` 读取，不再用 indoor3d 和 room_filelist。
- **模型**：`model_seam.py` 中最后一层输出 2 类，其余与 `sem_seg/model.py` 一致。
