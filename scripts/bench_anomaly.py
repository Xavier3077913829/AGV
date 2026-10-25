# -*- coding: utf-8 -*-
"""基准实验 1：设备异常检测 —— RF 基线 vs PyOD 多模型（生成报告用表格）

用法::

    python scripts/bench_anomaly.py
    python scripts/bench_anomaly.py --csv datasets/equipment/ai4i_equipment.csv
"""
import argparse
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "datasets", "equipment", "ai4i_equipment.csv"))
    ap.add_argument("--label", default=None,
                    help="标签列名；不填则自动识别（label / machine_failure / failure ...）")
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        sys.exit(f"找不到数据：{args.csv}")

    df = pd.read_csv(args.csv)
    print(f"数据：{os.path.basename(args.csv)}  {df.shape[0]} 行 × {df.shape[1]} 列")

    from EquipmentMonitoring.services_algo import (binarize_labels, compare_detectors,
                                                  find_label_column, pick_sensor_columns)
    label = args.label or find_label_column(df)
    if not label:
        sys.exit("未自动识别到标签列，请用 --label 指定")
    y = binarize_labels(df[label])
    print(f"标签列：{label}  原始取值：{df[label].unique().tolist()}")
    print(f"二值化后异常占比：{y.mean():.2%}（{int(y.sum())}/{len(y)}）\n")

    print("参与建模的传感器列：", pick_sensor_columns(df, label), "\n")

    table = compare_detectors(df, label_col=label)
    print("=== 检测器对比（测试集 30%）===")
    print(table.to_string(index=False))

    out = os.path.join(ROOT, "runs")
    os.makedirs(out, exist_ok=True)
    csv_out = os.path.join(out, "bench_anomaly.csv")
    table.to_csv(csv_out, index=False, encoding="utf-8-sig")
    print(f"\n已保存：{csv_out}")
    print("\n=== 结论（可直接写进报告）===")
    base = table[table["模型"].str.contains("基线")].iloc[0]
    unsup = table[~table["模型"].str.contains("基线")]
    best_u = unsup.iloc[0]
    print(f"  有监督基线 RandomForest：ROC-AUC={base['ROC-AUC']}, F1={base['F1']}")
    print(f"  最佳无监督检测器 {best_u['模型']}：ROC-AUC={best_u['ROC-AUC']}, F1={best_u['F1']}")
    gap = base["ROC-AUC"] - best_u["ROC-AUC"]
    print(f"  差距 {gap:+.4f}：有监督方法依赖标注，无监督方法无需标签，")
    print(f"  适合新设备冷启动/标签缺失场景；两者互补：线上先用无监督报警，再人工标注反哺 RF。")


if __name__ == "__main__":
    main()
