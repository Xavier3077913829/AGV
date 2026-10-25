# -*- coding: utf-8 -*-
"""把项目里的裂纹数据整理成 MVTec-AD 目录，供 anomalib 训练/评测。

用法::  python scripts/prepare_anomalib_dataset.py
输出::  datasets/auto/mt_crack/{train/good,test/crack,ground_truth/crack}
"""
import glob
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    base = os.path.join(ROOT, "datasets", "vision", "mt")
    good = sorted(glob.glob(os.path.join(base, "free", "images", "*")))
    crack = sorted(glob.glob(os.path.join(base, "crack", "images", "*")))
    masks = sorted(glob.glob(os.path.join(base, "crack", "masks", "*")))
    print(f"正常样本 {len(good)} 张 / 缺陷样本 {len(crack)} 张 / 掩膜 {len(masks)} 张")
    if not good or not crack:
        sys.exit("数据不足：请检查 datasets/vision/mt/{free,crack}/images")

    from ManufacturingVision.services_algo import build_mvtec_layout
    info = build_mvtec_layout(good, crack, masks,
                              os.path.join(ROOT, "datasets", "auto"))
    print("已生成 MVTec 风格数据集：")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("\n下一步：")
    print("  anomalib train --model Patchcore --data <生成的 root>   （或用 scripts/train_patchcore.py）")


if __name__ == "__main__":
    main()
