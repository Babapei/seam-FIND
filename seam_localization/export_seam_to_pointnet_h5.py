#!/usr/bin/env python3
"""
把 seam_localization 生成的点云分割数据导出为 pointnet-master/sem_seg 可读的 HDF5 格式。
导出后可在 pointnet-master 里直接用：改 NUM_CLASSES=2 和模型最后一层，用我们的 h5 训练。
格式：data (N, num_point, 9) 其中前 3 维为 xyz、后 6 维填 0；label (N, num_point) 0/1。
"""
import os
import sys
import argparse
import numpy as np
import h5py

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)

# pointnet-master 根目录（与 seam-FIND 同级的 pointnet-master 或项目内）
POINTNET_DIR = os.path.join(_root, "pointnet-master")
SEAM_DATA_DIR = os.path.join(_here, "pointcloud_data")
OUT_DIR = os.path.join(POINTNET_DIR, "sem_seg_seam_data")


def export_npy_to_h5(num_point=2048, num_features=9):
    """把 seam_train.npy / seam_val.npy 转为 pointnet sem_seg 的 h5。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    train_npy = os.path.join(SEAM_DATA_DIR, "seam_train.npy")
    val_npy = os.path.join(SEAM_DATA_DIR, "seam_val.npy")
    if not os.path.isfile(train_npy) or not os.path.isfile(val_npy):
        print("请先生成数据: python seam_localization/pointcloud_dataset.py --num_train 200 --num_val 50")
        sys.exit(1)

    for name, npy_path in [("train", train_npy), ("val", val_npy)]:
        data = np.load(npy_path, allow_pickle=True).item()
        points = data["points"]   # (N, P, 3)
        labels = data["labels"]   # (N, P)
        N, P, _ = points.shape
        if P != num_point:
            print("%s: 当前 P=%d，将裁剪/填充到 %d" % (name, P, num_point))
            if P >= num_point:
                points = points[:, :num_point, :]
                labels = labels[:, :num_point]
            else:
                pad = np.zeros((N, num_point - P, 3), dtype=points.dtype)
                points = np.concatenate([points, pad], axis=1)
                labels = np.concatenate([labels, np.zeros((N, num_point - P), dtype=labels.dtype)], axis=1)
        # pointnet sem_seg 输入是 9 维 (xyz + normal 等)，我们只有 xyz，后 6 维填 0
        data_9 = np.zeros((points.shape[0], points.shape[1], num_features), dtype=np.float32)
        data_9[:, :, :3] = points
        h5_path = os.path.join(OUT_DIR, "seam_%s.h5" % name)
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("data", data=data_9, compression="gzip", compression_opts=4)
            f.create_dataset("label", data=labels.astype(np.uint8), compression="gzip", compression_opts=1)
        print("已导出:", h5_path, "data shape:", data_9.shape, "label shape:", labels.shape)

    # 写文件列表，供 pointnet train 读取
    train_list = os.path.join(OUT_DIR, "train_files.txt")
    val_list = os.path.join(OUT_DIR, "val_files.txt")
    with open(train_list, "w") as f:
        f.write(os.path.join(OUT_DIR, "seam_train.h5") + "\n")
    with open(val_list, "w") as f:
        f.write(os.path.join(OUT_DIR, "seam_val.h5") + "\n")
    print("文件列表:", train_list, val_list)
    return OUT_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_point", type=int, default=2048)
    parser.add_argument("--num_features", type=int, default=9)
    args = parser.parse_args()
    export_npy_to_h5(args.num_point, args.num_features)
