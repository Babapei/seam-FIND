# seam-FIND / 焊缝寻位

> Weld seam localization from depth maps and point clouds: 2D traditional methods (depth valley / edge) and PointNet-based 3D segmentation for extracting seam trajectories.

基于深度图/点云的焊缝识别与 3D 轨迹提取，用于焊接机器人寻位。在 3D 视觉仿真尚未就绪时，用合成数据验证 2D 传统方法和 PointNet 深度学习两条路线。

## 主要内容

- **seam_localization/**：焊缝寻位核心模块
  - 2D 流派：深度图 → 2D 缝线（凹槽/边缘）→ 反投影 → 3D 轨迹
  - 3D 流派：点云 + PointNet 二类分割 → 缝点 → 3D 轨迹

- **method_compare**请具体看方法对比

## 快速运行

```bash
pip install requirements-seam.txt
python seam_localization/demo_run.py
```

Demo 使用合成深度图，输出 3D 轨迹和可视化图（`seam_localization/output/`）。

## 依赖

- numpy（必选）
- opencv-python、matplotlib（可选，用于部分方法与可视化）
- torch（若使用 PointNet 流派）

详见 `seam_localization/README.md` 和 `requirements-seam.txt`。

## 参考仓库

- pointnet-master、pointnet2-master、dgcnn-master、laser_line_extraction-master
