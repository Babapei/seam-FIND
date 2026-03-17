#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
焊缝寻位 Pipeline 统一训练入口
config 指定模型（pointnet|pointnet2|dgcnn），加载数据，训练，保存权重
"""
import os
import sys
import argparse
import yaml

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from seam_pipeline.models import get_model
from seam_pipeline.data import SeamDataset


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="seam_pipeline/config/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_name = cfg["model"]
    if model_name not in ("pointnet", "pointnet2", "dgcnn"):
        print("仅 pointnet/pointnet2/dgcnn 需训练。depth_valley/laser_line 无需训练。")
        sys.exit(0)

    data_cfg = cfg.get("data", {})
    train_npy = data_cfg.get("train_npy", "seam_localization/pointcloud_data/seam_train.npy")
    val_npy = data_cfg.get("val_npy", "seam_localization/pointcloud_data/seam_val.npy")
    num_points = data_cfg.get("num_points", 2048)

    train_path = os.path.join(_root, train_npy) if not os.path.isabs(train_npy) else train_npy
    val_path = os.path.join(_root, val_npy) if not os.path.isabs(val_npy) else val_npy
    if not os.path.isfile(train_path):
        print("训练数据不存在，请先运行: python seam_localization/pointcloud_dataset.py --num_train 200 --num_val 50")
        sys.exit(1)

    train_ds = SeamDataset(train_path, num_points=num_points)
    val_ds = SeamDataset(val_path, num_points=num_points) if os.path.isfile(val_path) else None

    train_cfg = cfg.get("train", {})
    batch_size = train_cfg.get("batch_size", 16)
    epochs = train_cfg.get("epochs", 50)
    lr = train_cfg.get("lr", 0.001)
    device = train_cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    device = torch.device(device)

    # 构建模型
    num_class = 2
    if model_name == "pointnet":
        m_cfg = cfg.get("pointnet", {})
        model = get_model("pointnet", num_class=m_cfg.get("num_class", 2), num_point=num_points)
    elif model_name == "pointnet2":
        m_cfg = cfg.get("pointnet2", {})
        model = get_model("pointnet2", num_class=m_cfg.get("num_class", 2), num_point=num_points)
    elif model_name == "dgcnn":
        m_cfg = cfg.get("dgcnn", {})
        model = get_model("dgcnn", num_class=m_cfg.get("num_class", 2), k=m_cfg.get("k", 20),
                         emb_dims=m_cfg.get("emb_dims", 1024), dropout=m_cfg.get("dropout", 0.5))
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=train_cfg.get("weight_decay", 1e-4))
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False) if val_ds else None

    out_cfg = cfg.get("output", {})
    ckpt_dir = out_cfg.get("checkpoint_dir", "seam_pipeline/checkpoints")
    exp_name = out_cfg.get("exp_name", "seam_exp")
    save_dir = os.path.join(_root, ckpt_dir, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    save_every = train_cfg.get("save_every", 5)

    print(f"Training {model_name} for {epochs} epochs, device={device}")
    for ep in range(epochs):
        model.train()
        total_loss = 0
        for pts, lab in train_loader:
            pts, lab = pts.to(device), lab.to(device)
            opt.zero_grad()
            logits = model(pts)
            loss = criterion(logits.reshape(-1, num_class), lab.reshape(-1))
            loss.backward()
            opt.step()
            total_loss += loss.item() * pts.size(0)
        avg = total_loss / len(train_ds)
        print(f"Epoch {ep+1}/{epochs} train_loss={avg:.4f}")

        if val_loader and (ep + 1) % 5 == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for pts, lab in val_loader:
                    pts, lab = pts.to(device), lab.to(device)
                    logits = model(pts)
                    pred = logits.argmax(2)
                    correct += (pred == lab).sum().item()
                    total += lab.numel()
            print(f"  val_acc={correct/total:.4f}")

        if (ep + 1) % save_every == 0:
            path = os.path.join(save_dir, f"{model_name}_{ep+1}.pt")
            torch.save({
                "model": model.state_dict(),
                "epoch": ep + 1,
                "num_point": num_points,
                "config": cfg,
            }, path)
            print(f"  saved {path}")

    final_path = os.path.join(save_dir, f"{model_name}_best.pt")
    torch.save({
        "model": model.state_dict(),
        "epoch": epochs,
        "num_point": num_points,
        "config": cfg,
    }, final_path)
    print(f"Done. Best saved: {final_path}")


if __name__ == "__main__":
    main()
