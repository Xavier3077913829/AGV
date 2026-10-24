"""Minimal DeepCrack training script.

Expected DeepCrack dataset layout:

    data_root/
        train_img/*.jpg
        train_lab/*.png   (0 background, 255 crack)
        test_img/*.jpg
        test_lab/*.png

Usage:
    python train.py --data_root /path/to/DeepCrack --save_path deepcrack.pth
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from deepcrack_model import DeepCrackNet


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p = torch.exp(-bce)
        loss = self.alpha * (1 - p) ** self.gamma * bce
        return loss.mean()


class CrackDataset(Dataset):
    def __init__(self, data_root: Path, split: str, image_size: int = 256, augment: bool = False):
        self.image_dir = data_root / f"{split}_img"
        self.label_dir = data_root / f"{split}_lab"
        self.image_size = image_size
        self.augment = augment
        self.samples = sorted(self.image_dir.glob("*.jpg"))
        if not self.samples:
            self.samples = sorted(self.image_dir.glob("*.png"))
        if not self.samples:
            raise FileNotFoundError(f"no images found in {self.image_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path = self.samples[index]
        label_path = self.label_dir / (image_path.stem + ".png")

        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path).convert("L")

        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        label = label.resize((self.image_size, self.image_size), Image.NEAREST)

        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                label = label.transpose(Image.FLIP_TOP_BOTTOM)

        image = np.asarray(image, dtype=np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = torch.from_numpy(image).permute(2, 0, 1)

        label = (np.asarray(label, dtype=np.float32) > 127).astype(np.float32)
        label = torch.from_numpy(label).unsqueeze(0)
        return image, label


def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred = (pred >= 0.5).astype(bool)
    target = target.astype(bool)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return float(intersection / max(union, 1))


def evaluate(model, loader, device):
    model.eval()
    total_iou = 0.0
    count = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            fused = model(images)[-1]
            probs = torch.sigmoid(fused).squeeze(1).cpu().numpy()
            targets = labels.squeeze(1).cpu().numpy()
            for prob, target in zip(probs, targets):
                total_iou += iou_score(prob, target)
                count += 1
    return total_iou / max(count, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DeepCrack on crack segmentation data")
    parser.add_argument("--data_root", required=True, help="DeepCrack dataset root with *_img and *_lab folders")
    parser.add_argument("--save_path", default="deepcrack.pth")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--step_every", type=int, default=40, help="multiply LR by 0.1 every N epochs")
    parser.add_argument("--val_every", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _save(model, path):
    """保存权重（父目录不存在则创建）。"""
    import os as _os
    _os.makedirs(_os.path.dirname(_os.path.abspath(path)) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        print("[warn] 当前 PyTorch 为 CPU 版本，--device cuda 不可用，自动改用 cpu")
        args.device = "cpu"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    import os as _os0
    _os0.makedirs(_os0.path.dirname(_os0.path.abspath(args.save_path)) or ".", exist_ok=True)
    data_root = Path(args.data_root)

    train_set = CrackDataset(data_root, "train", args.image_size, augment=True)
    test_set = CrackDataset(data_root, "test", args.image_size, augment=False)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = DeepCrackNet(in_channels=3, num_classes=1, base_filters=64, norm="batch").to(device)
    criterion = BinaryFocalLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=2e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.step_every, gamma=0.1
    )
    side_weights = [0.5, 0.75, 1.0, 0.75, 0.5]

    print(f"train samples: {len(train_set)}, test samples: {len(test_set)}, device: {device}")
    best_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            loss = 0.0
            for side_output, weight in zip(outputs[:-1], side_weights):
                loss = loss + weight * criterion(side_output, labels)
            loss = loss + criterion(outputs[-1], labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)

        scheduler.step()
        epoch_loss /= len(train_set)
        print(f"epoch {epoch:03d}  loss {epoch_loss:.4f}  lr {scheduler.get_last_lr()[0]:.2e}")

        if epoch % args.val_every == 0:
            iou = evaluate(model, test_loader, device)
            print(f"  test IoU {iou:.4f}")
            if iou > best_iou:
                best_iou = iou
                _save(model, args.save_path)
                print(f"  saved best checkpoint to {args.save_path}")

    if best_iou == 0.0:
        _save(model, args.save_path)
        print(f"saved final checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
