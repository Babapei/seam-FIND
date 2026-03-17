# 实验结果

```
=== seam_pipeline 测试 ===
模型: pointnet | 测试数: 20

--- 指标 ---
accuracy:  0.9845
iou_seam:  0.5421
iou_bg:    0.9842
mean_iou:  0.7631
指标已保存: C:\Users\xuwangweifan\Desktop\项目\seam-FIND\seam_pipeline/output/pointnet_test\test_metrics.txt
  可视化已保存: C:\Users\xuwangweifan\Desktop\项目\seam-FIND\seam_pipeline/output/pointnet_test\test_viz_sample_0.png
  可视化已保存: C:\Users\xuwangweifan\Desktop\项目\seam-FIND\seam_pipeline/output/pointnet_test\test_viz_sample_1.png
  可视化已保存: C:\Users\xuwangweifan\Desktop\项目\seam-FIND\seam_pipeline/output/pointnet_test\test_viz_sample_2.png

=== seam_pipeline 2D 测试 ===
提取器: depth_valley | 测试数: 20

--- 指标 ---
avg_dist_pred2gt_m:  0.005365
avg_dist_gt2pred_m:  0.003904
mean_traj_dist_m:    0.004634
指标已保存: C:\Users\xuwangweifan\Desktop\项目\seam-FIND\seam_pipeline/output/dv_test\test_2d_metrics.txt
  可视化已保存: C:\Users\xuwangweifan\Desktop\项目\seam-FIND\seam_pipeline/output/dv_test\test_2d_viz_sample_0.png
  可视化已保存: C:\Users\xuwangweifan\Desktop\项目\seam-FIND\seam_pipeline/output/dv_test\test_2d_viz_sample_1.png
  可视化已保存: C:\Users\xuwangweifan\Desktop\项目\seam-FIND\seam_pipeline/output/dv_test\test_2d_viz_sample_2.png

=== seam_pipeline 2D 测试 ===
提取器: laser_line | 测试数: 20

--- 指标 ---
avg_dist_pred2gt_m:  0.009346
avg_dist_gt2pred_m:  0.010013
mean_traj_dist_m:    0.009679
指标已保存: C:\Users\xuwangweifan\Desktop\项目\seam-FIND\seam_pipeline/output/ll_test\test_2d_metrics.txt
  可视化已保存: C:\Users\xuwangweifan\Desktop\项目\seam-FIND\seam_pipeline/output/ll_test\test_2d_viz_sample_0.png
  可视化已保存: C:\Users\xuwangweifan\Desktop\项目\seam-FIND\seam_pipeline/output/ll_test\test_2d_viz_sample_1.png
  可视化已保存: C:\Users\xuwangweifan\Desktop\项目\seam-FIND\seam_pipeline/output/ll_test\test_2d_viz_sample_2.png

```
# 解释

## 一、PointNet（点云深度学习）

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| **accuracy** | 整体分类准确率 | 逐点比较 `pred == labels`，正确点数 / 总点数 |
| **iou_seam** | 缝点（class 1）的 IoU | `交集 / 并集`，交集 = 预测为缝且 GT 为缝的点，并集 = 预测或 GT 为缝的点 |
| **iou_bg** | 背景点（class 0）的 IoU | 同上，针对背景类 |
| **mean_iou** | 两类的平均 IoU | `(iou_seam + iou_bg) / 2` |

### 结果解读

- **accuracy 0.9845**：约 98.45% 的点分类正确，整体很高。
- **iou_bg 0.9842**：背景分割很准，背景 IoU 接近 1。
- **iou_seam 0.5421**：缝点 IoU 明显偏低，说明缝区域存在漏检/多检或边界不精确。
- **mean_iou 0.7631**：被 iou_seam 拉低。

在焊缝点云中，缝点占比通常很小（几十到几百点 vs 上千背景点），accuracy 容易偏高；**iou_seam** 更能反映缝本身的检测质量，0.54 表示缝分割还有提升空间。

---

## 二、2D 方法（depth_valley / laser_line）

这些指标基于**预测轨迹 vs GT 轨迹**的空间距离：

| 指标 | 含义 | 计算方式 |
|------|------|----------|
| **avg_dist_pred2gt_m** | 预测 → GT 的平均距离 | 对每个预测点，找最近的 GT 点，算距离，再取平均 |
| **avg_dist_gt2pred_m** | GT → 预测的平均距离 | 对每个 GT 点，找最近的预测点，算距离，再取平均 |
| **mean_traj_dist_m** | 轨迹级平均偏差 | `(pred2gt + gt2pred) / 2`，单位米 |

### 单位

- 单位均为**米（m）**，表示预测轨迹与 GT 轨迹之间的物理偏差。

### 结果解读

| 方法 | pred2gt | gt2pred | mean_traj | 含义 |
|------|---------|---------|-----------|------|
| **depth_valley** | 0.0054 m | 0.0039 m | 0.0046 m | 轨迹平均偏差约 **4.6 mm** |
| **laser_line** | 0.0093 m | 0.0100 m | 0.0097 m | 轨迹平均偏差约 **9.7 mm** |

- **pred2gt > gt2pred**：预测点更稀疏或偏少，每个预测点到 GT 的平均距离更大。
- **pred2gt < gt2pred**：预测点更密集或偏多，GT 点到最近预测点的平均距离更大。
- **depth_valley** 偏差更小，在当前合成深度图 + 凹槽缝下表现更好。
- **laser_line** 偏差约为 depth_valley 的两倍，与算法简化、取“最接近行中心的线段”等设定有关，且当前数据形态未必最适合 laser_line。

---

## 总结

| 方法 | 主要指标 | 粗略结论 |
|------|----------|----------|
| **PointNet** | accuracy、iou_seam、mean_iou | 分类很准（~98%），缝 IoU 中等（~54%），缝区域分割仍需优化 |
| **depth_valley** | mean_traj_dist_m | 轨迹误差约 4.6 mm，2D 方法中效果最好 |
| **laser_line** | mean_traj_dist_m | 轨迹误差约 9.7 mm，略逊于 depth_valley |