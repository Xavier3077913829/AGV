# -*- coding: utf-8 -*-
"""PatchCore（anomalib 2.x）训练 + 评估，输出图像级 AUROC / F1

用法::

    python scripts/run_patchcore.py                # 训练 + 测试
    python scripts/run_patchcore.py --epochs 1     # 默认 1（PatchCore 无需多轮）

假设数据已由 scripts/prepare_anomalib_dataset.py 转好：
    datasets/auto/mt_crack/{train/good, test/crack, ground_truth/crack}
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "datasets", "auto", "mt_crack"))
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "patchcore"))
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--coreset", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    if not os.path.isdir(os.path.join(args.data, "train", "good")):
        sys.exit("找不到 datasets/auto/mt_crack/train/good —— 先跑 scripts/prepare_anomalib_dataset.py")

    try:
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from anomalib.models import Patchcore
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"anomalib 不可用：{type(exc).__name__}: {exc}")

    os.makedirs(args.out, exist_ok=True)

    datamodule = Folder(
        name="mt_crack",
        root=args.data,
        normal_dir="train/good",
        abnormal_dir="test/crack",
        mask_dir="ground_truth/crack",
        train_batch_size=args.batch,
        eval_batch_size=args.batch,
        num_workers=0,
    )

    model = Patchcore(backbone=args.backbone, coreset_sampling_ratio=args.coreset)
    engine = Engine(default_root_dir=args.out, max_epochs=args.epochs)

    print(f"[1/2] 训练 PatchCore（backbone={args.backbone}, coreset={args.coreset}）...")
    engine.fit(model=model, datamodule=datamodule)
    print("      训练完成（PatchCore 只建特征记忆库，无梯度训练）")

    print("[2/2] 在裂纹测试集上评估 ...")
    results = engine.test(model=model, datamodule=datamodule)

    # results 可能是 dict / list[dict]，统一取第一个
    metrics = results[0] if isinstance(results, (list, tuple)) and results else results
    print("\n=== PatchCore 测试结果（图像级）===")
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                print(f"  {k:<44} {v:.4f}")
    else:
        print("  ", metrics)

    # 保存一份纯文本报告
    report = os.path.join(ROOT, "runs", "patchcore_metrics.txt")
    with open(report, "w", encoding="utf-8") as fh:
        fh.write("PatchCore 评估（anomalib 2.x, CPU）\n")
        fh.write(f"数据: {args.data}\n正常样本: 100  缺陷样本: 57\n\n")
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    fh.write(f"{k}: {v:.4f}\n")
    print(f"\n指标已保存：{report}")


if __name__ == "__main__":
    main()
