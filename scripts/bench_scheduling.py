# -*- coding: utf-8 -*-
"""调度算法基准：基线（按优先级最早完成）vs 最优插入 + 变邻域下降（VND）

算例说明
--------
- 默认用 **Solomon 风格的合成算例**（100×100 平面、depot 在中心、客户成簇分布、
  固定随机种子可复现）：客户坐标→地图节点，客户需求→货物重量，
  AGV 数量/载重按 Solomon 惯例设置。
- 支持 ``--solomon <file>`` 读取**真实 Solomon CVRPTW 实例文件**
  （格式：NODE 段 + DEMAND 段 + 可选 DEPOT 段）。
- ⚠️ 本项目调度模型**没有客户时间窗**（只有节点避碰预约），因此 Solomon 的
  时间窗字段不参与计算；报告里请写"Solomon 风格算例（坐标与需求维度一致，时间窗未建模）"。

用法::

    python scripts/bench_scheduling.py                 # 20 客户 / 3 台 AGV
    python scripts/bench_scheduling.py -n 30 -k 4      # 30 客户 / 4 台 AGV
    python scripts/bench_scheduling.py --solomon C101.txt
"""
from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def parse_solomon(path: str):
    """解析 Solomon CVRPTW 实例文件，返回 (name, depot, customers, capacity, vehicles)。

    customers: list[dict(x, y, demand, ready, due)]
    """
    with open(path, encoding="utf-8", errors="ignore") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    name = lines[0] if lines else "solomon"
    capacity, vehicles = 100.0, 25
    i = 0
    while i < len(lines):
        if lines[i].upper().startswith("VEHICLE"):
            parts = lines[i + 1].split()
            if len(parts) >= 2:
                vehicles, capacity = int(parts[0]), float(parts[1])
            i += 2
            continue
        if lines[i].upper().startswith("NODE"):
            i += 1
            depot = None
            customers = []
            while i < len(lines) and not lines[i].upper().startswith("DEMAND"):
                parts = lines[i].split()
                if len(parts) >= 5:
                    idx = int(parts[0])
                    rec = {"id": idx, "x": float(parts[1]), "y": float(parts[2]),
                           "ready": float(parts[3]), "due": float(parts[4]),
                           "demand": 0.0}
                    if idx == 0:
                        depot = rec
                    else:
                        customers.append(rec)
                i += 1
            # DEMAND 段
            i += 1
            di = 0
            while i < len(lines) and di < len(customers):
                parts = lines[i].split()
                if parts:
                    customers[di]["demand"] = float(parts[-1])
                    di += 1
                i += 1
            return name, depot, customers, capacity, vehicles
        i += 1
    raise SystemExit(f"无法解析 Solomon 文件：{path}")


def build_instance(n_customers=20, n_agvs=3, seed=42, clustered=True):
    """构造 Solomon 风格算例：返回 (nodes, graph, agvs, tasks, meta)。"""
    from SchedulingDecision.models import AGV, MapNode, TransportTask

    rng = random.Random(seed)
    # depot 在 (50,50)，客户成簇分布在 100×100 平面
    nodes = []
    depot = MapNode(code="D0", name="Depot", x=50.0, y=50.0)
    depot.pk = 1
    nodes.append(depot)

    cluster_centers = [(30, 30), (70, 30), (30, 70), (70, 70)] if clustered else [(50, 50)]
    for i in range(1, n_customers + 1):
        cx, cy = cluster_centers[i % len(cluster_centers)]
        x = max(0.0, min(100.0, cx + rng.uniform(-12, 12)))
        y = max(0.0, min(100.0, cy + rng.uniform(-12, 12)))
        node = MapNode(code=f"C{i}", name=f"客户{i}", x=round(x, 2), y=round(y, 2))
        node.pk = i + 1
        nodes.append(node)

    # 完全图（欧氏距离）
    import math
    graph = {}
    for a in nodes:
        graph[a.pk] = [
            (b.pk, round(math.dist((a.x, a.y), (b.x, b.y)), 3), 1.0)
            for b in nodes if b.pk != a.pk
        ]

    agvs = []
    for k in range(n_agvs):
        agv = AGV(code=f"AGV{k+1}", name=f"AGV{k+1}", speed=1.0,
                  battery_percent=100.0, payload_capacity=100.0)
        agv.pk = k + 1
        agv.current_node_id = depot.pk
        agvs.append(agv)

    tasks = []
    for i, node in enumerate(nodes[1:], start=1):
        task = TransportTask(task_no=f"T{i:03d}", cargo_name=f"货物{i}",
                             cargo_weight=round(rng.uniform(5, 40), 1), priority=2)
        task.pk = i
        task.pickup_node_id = depot.pk     # depot 取货
        task.dropoff_node_id = node.pk     # 送到客户
        tasks.append(task)

    meta = {
        "customers": n_customers,
        "agvs": n_agvs,
        "area": "100x100（Solomon 惯例）",
        "seed": seed,
        "clustered": clustered,
    }
    return nodes, graph, agvs, tasks, meta


def build_solomon_instance(path):
    """把真实 Solomon 实例转成项目模型对象。"""
    from SchedulingDecision.models import AGV, MapNode, TransportTask

    name, depot_rec, customers, capacity, vehicles = parse_solomon(path)
    nodes = []
    depot = MapNode(code="D0", name="Depot", x=depot_rec["x"], y=depot_rec["y"])
    depot.pk = 1
    nodes.append(depot)
    for i, c in enumerate(customers, start=1):
        node = MapNode(code=f"C{i}", name=f"客户{c['id']}", x=c["x"], y=c["y"])
        node.pk = i + 1
        nodes.append(node)

    import math
    graph = {}
    for a in nodes:
        graph[a.pk] = [(b.pk, round(math.dist((a.x, a.y), (b.x, b.y)), 3), 1.0)
                       for b in nodes if b.pk != a.pk]

    agvs = []
    for k in range(max(1, vehicles)):
        agv = AGV(code=f"AGV{k+1}", name=f"AGV{k+1}", speed=1.0,
                  battery_percent=100.0, payload_capacity=capacity)
        agv.pk = k + 1
        agv.current_node_id = depot.pk
        agvs.append(agv)

    tasks = []
    for i, (c, node) in enumerate(zip(customers, nodes[1:]), start=1):
        task = TransportTask(task_no=f"T{i:03d}", cargo_name=f"货物{i}",
                             cargo_weight=max(0.1, c["demand"]), priority=2)
        task.pk = i
        task.pickup_node_id = depot.pk
        task.dropoff_node_id = node.pk
        tasks.append(task)

    meta = {"instance": name, "customers": len(customers), "agvs": vehicles,
            "capacity": capacity, "file": os.path.basename(path)}
    return nodes, graph, agvs, tasks, meta


def run_once(label, builder, tasks, agvs, router, now, weights):
    t0 = time.perf_counter()
    result = builder(tasks, agvs, router, now, weights)
    elapsed = time.perf_counter() - t0
    # optimize_multi_solution 返回 (最优解, 基线解, 评估次数)
    evaluation = result[0] if isinstance(result, tuple) else result
    return {
        "算法": label,
        "可行": "是" if evaluation.feasible else "否",
        "目标值": round(evaluation.objective, 2),
        "makespan(s)": round(evaluation.makespan_seconds, 1),
        "总距离(m)": round(evaluation.total_distance, 1),
        "等待(s)": round(evaluation.total_wait_seconds, 1),
        "用车": evaluation.used_agvs,
        "耗时(s)": round(elapsed, 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--customers", type=int, default=20)
    ap.add_argument("-k", "--agvs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--solomon", default="", help="Solomon 实例文件路径")
    args = ap.parse_args()

    # Django 环境（需要 ORM 模型定义，不会写数据库）
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "AGV.settings")
    import django
    django.setup()

    from SchedulingDecision.services import (
        GraphRouter,
        build_baseline_solution,
        optimize_multi_solution,
    )

    if args.solomon:
        nodes, graph, agvs, tasks, meta = build_solomon_instance(args.solomon)
    else:
        nodes, graph, agvs, tasks, meta = build_instance(args.customers, args.agvs, args.seed)

    router = GraphRouter(graph)
    now = datetime(2026, 10, 19, 8, 0, 0)

    print("=" * 78)
    print("调度算法基准（Solomon 风格算例）")
    for k, v in meta.items():
        print(f"  {k}: {v}")
    print(f"  任务数: {len(tasks)}   AGV 数: {len(agvs)}")

    rows = [
        run_once("基线：按优先级取最早完成", build_baseline_solution, tasks, agvs, router, now, None),
        run_once("优化：最优插入 + VND", optimize_multi_solution, tasks, agvs, router, now, None),
    ]

    print("\n" + "=" * 78)
    header = list(rows[0].keys())
    print("  " + "".join(f"{h:<14}" for h in header))
    for r in rows:
        print("  " + "".join(f"{str(r[h]):<14}" for h in header))

    base, opt = rows[0], rows[1]
    if base["目标值"]:
        imp = (base["目标值"] - opt["目标值"]) / base["目标值"] * 100
        print(f"\n目标值改进：{imp:+.2f}%    "
              f"makespan：{base['makespan(s)']} → {opt['makespan(s)']} s    "
              f"总距离：{base['总距离(m)']} → {opt['总距离(m)']} m")
        print(f"耗时：{base['耗时(s)']} → {opt['耗时(s)']} s（仍在秒级内）")

    out = os.path.join(ROOT, "runs")
    os.makedirs(out, exist_ok=True)
    import csv
    path = os.path.join(out, "bench_scheduling.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n已保存：{path}")
    print("\n报告写法示例：")
    print(f"  在 {meta.get('customers')} 个客户 / {meta.get('agvs')} 台 AGV 的 Solomon 风格算例上，"
          f"最优插入 + 变邻域下降相对基线将目标值降低 {imp:.1f}%，求解时间 {opt['耗时(s)']} s。")


if __name__ == "__main__":
    main()
