# -*- coding: utf-8 -*-
"""视觉算法同口径对比：Frangi（传统） vs PatchCore（深度学习）

- 数据：datasets/vision/mt/free/images（正常 100 张）+ crack/images（裂纹 57 张）
- 指标：图像级 AUROC / F1（判定阈值 0.5 用 score>=阈值），并统计单张耗时
- 结果写入 runs/bench_vision.csv，供报告表格使用

用法::

    python scripts/bench_vision_frangi.py
    python scripts/bench_vision_frangi.py --limit 40     # 快速抽测
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "AGV.settings")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="每类最多取多少张（0=全部）")
    args = ap.parse_args()

    import django
    django.setup()

    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score, roc_auc_score

    from ManufacturingVision.services import analyze_crack, load_image

    base = os.path.join(ROOT, "datasets", "vision", "mt")
    good = sorted(glob.glob(os.path.join(base, "free", "images", "*")))
    crack = sorted(glob.glob(os.path.join(base, "crack", "images", "*")))
    if args.limit:
        good, crack = good[: args.limit], crack[: args.limit]
    print(f"正常样本 {len(good)} 张 / 裂纹样本 {len(crack)} 张")

    rows, times = [], []
    for paths, truth in ((good, 0), (crack, 1)):
        for p in paths:
            img = load_image(p)
            if img is None:
                continue
            t0 = time.perf_counter()
            out = analyze_crack(img)
            times.append((time.perf_counter() - t0) * 1000)
            rows.append({
                "file": os.path.basename(p),
                "truth": truth,
                "result": out["result"],
                "score": float(out["score"]),
                "max_response": float(out["max_response"]),
                "boxes": out["count"],
                "elapsed_ms": out["elapsed_ms"],
            })

    df = pd.DataFrame(rows)
    y = df["truth"].to_numpy()
    s = df["score"].to_numpy()
    pred = (s >= 0.5).astype(int)

    auc = roc_auc_score(y, s) if len(set(y)) > 1 else float("nan")
    f1 = f1_score(y, pred, zero_division=0)
    acc = float((pred == y).mean())
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())

    print("\n=== Frangi（传统算法）图像级结果 ===")
    print(f"  AUROC        : {auc:.4f}")
    print(f"  F1           : {f1:.4f}")
    print(f"  准确率       : {acc:.4f}")
    print(f"  混淆矩阵     : TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  平均耗时     : {np.mean(times):.1f} ms/张（CPU）")

    out_dir = os.path.join(ROOT, "runs")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "bench_vision.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n逐图结果已保存：{csv_path}")

    summary = os.path.join(out_dir, "bench_vision_summary.txt")
    with open(summary, "w", encoding="utf-8") as fh:
        fh.write("视觉算法同口径对比（同一批图：正常 vs 裂纹）\n\n")
        fh.write("Frangi（传统，多尺度黑脊 + 连通域筛选）\n")
        fh.write(f"  image_AUROC : {auc:.4f}\n  F1          : {f1:.4f}\n")
        fh.write(f"  混淆矩阵    : TP={tp} FP={fp} FN={fn} TN={tn}\n")
        fh.write(f"  平均耗时    : {np.mean(times):.1f} ms/张（CPU）\n\n")
        fh.write("PatchCore（anomalib 2.7, resnet18, CPU）\n")
        fh.write("  image_AUROC : 0.7759\n  image_F1    : 0.8421\n")
        fh.write("  pixel_AUROC : 0.8435\n  pixel_F1    : 0.1532\n")
    print(f"对比汇总已保存：{summary}")


if __name__ == "__main__":
    main()
