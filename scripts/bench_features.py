# -*- coding: utf-8 -*-
"""基准实验 2：tsfresh 自动特征提取（特征数 / 耗时）

用法::  python scripts/bench_features.py
"""
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    csv = os.path.join(ROOT, "datasets", "equipment", "ai4i_equipment.csv")
    df = pd.read_csv(csv).head(2000)
    print(f"数据：{os.path.basename(csv)}  取前 {len(df)} 行做窗口切分")

    from EquipmentMonitoring.services_algo import extract_tsfresh_features
    t0 = time.time()
    try:
        feats = extract_tsfresh_features(df, window=64)
    except RuntimeError as e:
        sys.exit(f"[需要安装] {e}")
    dt = time.time() - t0

    print(f"\n窗口数：{len(feats)}   自动生成特征：{feats.shape[1]} 个   耗时：{dt:.2f} s")
    print("样例特征列（前 12）：", list(feats.columns)[:12])

    out = os.path.join(ROOT, "runs")
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, "tsfresh_features.csv")
    feats.to_csv(p, encoding="utf-8-sig")
    print(f"\n已保存特征矩阵：{p}")
    print("\n报告写法示例：")
    print(f"  使用 tsfresh 对 64 点滑窗自动提取 {feats.shape[1]} 个特征，"
          f"较手工特征工程更全面地刻画了振动/温度/扭矩的统计与形态特性。")


if __name__ == "__main__":
    main()
