#!/usr/bin/env python3
# 训练 PointNet 焊缝二类分割。先运行 pointcloud_dataset 生成数据。
import os
import sys
import argparse

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from seam_localization.pointnet_seam_model import PointNetSeg
from seam_localization.pointnet_seam_data import SeamPointCloudDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_points", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = os.path.join(_here, "pointcloud_data")
    train_npy = os.path.join(args.data_dir, "seam_train.npy")
    val_npy = os.path.join(args.data_dir, "seam_val.npy")
    if not os.path.isfile(train_npy) or not os.path.isfile(val_npy):
        print("未找到数据，请先运行:")
        print("  python seam_localization/pointcloud_dataset.py --num_train 200 --num_val 50")
        sys.exit(1)

    if args.save_dir is None:
        args.save_dir = os.path.join(_here, "checkpoints")
    os.makedirs(args.save_dir, exist_ok=True)

    train_ds = SeamPointCloudDataset(train_npy, num_points=args.num_points, train=True)
    val_ds = SeamPointCloudDataset(val_npy, num_points=args.num_points, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = PointNetSeg(num_class=2, num_point=args.num_points).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for pts, lab in train_loader:
            pts, lab = pts.to(args.device), lab.to(args.device)
            opt.zero_grad()
            logits = model(pts)
            loss = criterion(logits.reshape(-1, 2), lab.reshape(-1))
            loss.backward()
            opt.step()
            total_loss += loss.item()
            pred = logits.argmax(2)
            correct += (pred == lab).sum().item()
            total += lab.numel()
        train_acc = correct / total
        print("Epoch %d train loss %.4f acc %.4f" % (epoch + 1, total_loss / len(train_loader), train_acc))

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for pts, lab in val_loader:
                pts, lab = pts.to(args.device), lab.to(args.device)
                logits = model(pts)
                pred = logits.argmax(2)
                correct += (pred == lab).sum().item()
                total += lab.numel()
        val_acc = correct / total
        print("         val acc %.4f" % val_acc)

    ckpt = os.path.join(args.save_dir, "pointnet_seam.pt")
    torch.save({"model": model.state_dict(), "num_class": 2, "num_point": args.num_points}, ckpt)
    print("Saved:", ckpt)


if __name__ == "__main__":
    main()
