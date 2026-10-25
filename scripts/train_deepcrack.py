# -*- coding: utf-8 -*-
"""训练 DeepCrack（监督式裂纹分割）

数据：datasets/vision/deepcrack/{train_img,train_lab,test_img,test_lab}
权重输出：runs/deepcrack/deepcrack.pth

用法::

    python scripts/train_deepcrack.py --epochs 20 --batch 4            # CPU 小规模
    python scripts/train_deepcrack.py --epochs 200 --batch 8 --device cuda
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "datasets", "vision", "deepcrack")
OUT = os.path.join(ROOT, "runs", "deepcrack")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if not os.path.isdir(os.path.join(DATA, "train_img")):
        sys.exit("找不到 datasets/vision/deepcrack/train_img")

    os.makedirs(OUT, exist_ok=True)
    train_py = os.path.join(ROOT, "scripts", "_deepcrack_train_inner.py")
    inner = os.path.join(ROOT, "ManufacturingVision", "deepcrack_train.py")
    if not os.path.exists(inner):
        sys.exit("缺少 ManufacturingVision/deepcrack_train.py（训练实现）")

    cmd = [sys.executable, inner,
           "--data_root", DATA,
           "--save_path", os.path.join(OUT, "deepcrack.pth"),
           "--epochs", str(args.epochs),
           "--batch_size", str(args.batch),
           "--image_size", str(args.size),
           "--lr", str(args.lr),
           "--device", args.device]
    print("执行：", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
