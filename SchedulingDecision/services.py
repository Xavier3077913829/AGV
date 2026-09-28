from __future__ import annotations

import copy
import heapq
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable

from django.db import transaction
from django.utils import timezone

from .models import AGV, DispatchRecord, MapEdge, MapNode, ScheduleRun, TransportTask


MIN_BATTERY_PERCENT = 15.0
SAFETY_GAP_SECONDS = 2.0
LOADING_SECONDS = 6.0
UNLOADING_SECONDS = 6.0
BATTERY_DROP_PER_METER = 0.06
DEFAULT_AGV_WEIGHT = 30.0
DEFAULT_WAIT_WEIGHT = 0.20
DEFAULT_DISTANCE_WEIGHT = 0.005
DEFAULT_PRIORITY_WEIGHT = 0.02


class SchedulingError(Exception):
    """调度请求无法完成时抛出。"""


@dataclass
class PathResult:
    nodes: list[int]
    distance: float


@dataclass
class TaskPlan:
    task_id: int
    task_no: str
    agv_id: int
    agv_code: str
    sequence_number: int
    start_time: datetime
    end_time: datetime
    route: dict
    distance: float
    wait_seconds: float
    duration_seconds: float


@dataclass
class ScheduleEvaluation:
    feasible: bool
    objective: float = 0.0
    makespan_seconds: float = 0.0
    total_completion_seconds: float = 0.0
    total_distance: float = 0.0
    total_wait_seconds: float = 0.0
    used_agvs: int = 0
    assignments: dict[int, int] = field(default_factory=dict)
    orders: dict[int, list[int]] = field(default_factory=dict)
    plans: dict[int, TaskPlan] = field(default_factory=dict)
    agv_summary: dict[int, dict] = field(default_factory=dict)
    message: str = ""

def build_graph() -> dict[int, list[tuple[int, float, float]]]:
    """构建仓库地图邻接表：node -> [(neighbor, distance, speed_limit)]."""
    adjacency: dict[int, list[tuple[int, float, float]]] = {}
    for node in MapNode.objects.filter(is_active=True):
        adjacency[node.id] = []

    edges = MapEdge.objects.filter(
        is_active=True,
        start_node__is_active=True,
        end_node__is_active=True,
    ).select_related("start_node", "end_node")
    for edge in edges:
        adjacency.setdefault(edge.start_node_id, []).append(
            (edge.end_node_id, edge.distance, edge.speed_limit)
        )
        if edge.bidirectional:
            adjacency.setdefault(edge.end_node_id, []).append(
                (edge.start_node_id, edge.distance, edge.speed_limit)
            )
    return adjacency


class GraphRouter:
    """带最短路径缓存的 Dijkstra 路由器。"""

    def __init__(self, graph: dict[int, list[tuple[int, float, float]]]):
        self.graph = graph
        self.cache: dict[tuple[int, int], PathResult] = {}

    def path(self, start: int, end: int) -> PathResult:
        key = (start, end)
        if key not in self.cache:
            self.cache[key] = self._dijkstra(start, end)
        return self.cache[key]

    def _dijkstra(self, start: int, end: int) -> PathResult:
        if start == end:
            return PathResult([start], 0.0)
        if start not in self.graph or end not in self.graph:
            raise SchedulingError(f"地图节点 {start} 或 {end} 不可达。")

        distances = {start: 0.0}
        previous: dict[int, int] = {}
        queue = [(0.0, start)]
        visited = set()

        while queue:
            current_distance, current = heapq.heappop(queue)
            if current in visited:
                continue
            visited.add(current)
            if current == end:
                break
            for neighbor, edge_distance, _speed_limit in self.graph.get(current, []):
                next_distance = current_distance + edge_distance
                if next_distance < distances.get(neighbor, float("inf")):
                    distances[neighbor] = next_distance
                    previous[neighbor] = current
                    heapq.heappush(queue, (next_distance, neighbor))

        if end not in distances:
            raise SchedulingError("起点和终点之间没有可用路径。")

        nodes = [end]
        while nodes[-1] != start:
            nodes.append(previous[nodes[-1]])
        nodes.reverse()
        return PathResult(nodes, round(distances[end], 3))


def shortest_path(graph, start: int, end: int) -> PathResult:
    """兼容旧调用的最短路径函数。"""
    return GraphRouter(graph).path(start, end)


def _join_paths(first: PathResult, second: PathResult) -> PathResult:
    if first.nodes and second.nodes and first.nodes[-1] == second.nodes[0]:
        nodes = first.nodes + second.nodes[1:]
    else:
        nodes = first.nodes + second.nodes
    return PathResult(nodes, round(first.distance + second.distance, 3))


def _segment_distance(graph, start: int, end: int) -> float:
    candidates = [item[1] for item in graph.get(start, []) if item[0] == end]
    return min(candidates) if candidates else 0.1

def _resolve_node_wait(
    reservations: dict[int, list[dict]],
    node_id: int,
    arrival: datetime,
    agv_id: int,
    task_id: int,
) -> float:
    proposed_start = arrival
    proposed_end = arrival + timedelta(seconds=SAFETY_GAP_SECONDS)
    for interval in sorted(reservations.get(node_id, []), key=lambda item: item["start"]):
        if interval["agv_id"] == agv_id and interval["task_id"] == task_id:
            continue
        overlaps = proposed_start <= interval["end"] and proposed_end >= interval["start"]
        if overlaps:
            proposed_start = max(
                proposed_start, interval["end"] + timedelta(seconds=SAFETY_GAP_SECONDS)
            )
            proposed_end = proposed_start + timedelta(seconds=SAFETY_GAP_SECONDS)
    return max(0.0, (proposed_start - arrival).total_seconds())


def _reserve_node(
    reservations: dict[int, list[dict]],
    node_id: int,
    start: datetime,
    end: datetime,
    agv_id: int,
    task_id: int,
) -> None:
    reservations.setdefault(node_id, []).append(
        {"start": start, "end": end, "agv_id": agv_id, "task_id": task_id}
    )


def _plan_task_route(
    agv: AGV,
    task: TransportTask,
    router: GraphRouter,
    base_time: datetime,
    reservations: dict[int, list[dict]],
    start_node_id: int | None = None,
) -> tuple[dict, datetime, dict[int, list[dict]], float, float]:
    """规划单任务路线，并返回路线、完成时间、预约表、距离和等待时间。"""
    start_node = start_node_id if start_node_id is not None else agv.current_node_id
    to_pickup = router.path(start_node, task.pickup_node_id)
    to_dropoff = router.path(task.pickup_node_id, task.dropoff_node_id)
    path = _join_paths(to_pickup, to_dropoff)
    speed = max(agv.speed, 0.1)
    cursor = base_time
    current_node = path.nodes[0]
    timeline = []
    waits = []
    legs = []
    updated_reservations = copy.deepcopy(reservations)

    start_wait = _resolve_node_wait(
        updated_reservations, current_node, cursor, agv.id, task.id
    )
    if start_wait > 0:
        waits.append({
            "node_id": current_node,
            "seconds": round(start_wait, 2),
            "reason": "始发避碰等待",
        })
        cursor += timedelta(seconds=start_wait)
    _reserve_node(
        updated_reservations,
        current_node,
        cursor,
        cursor + timedelta(seconds=SAFETY_GAP_SECONDS),
        agv.id,
        task.id,
    )
    timeline.append({
        "node_id": current_node,
        "arrival_seconds": 0.0,
        "departure_seconds": round(start_wait + SAFETY_GAP_SECONDS, 2),
        "wait_seconds": round(start_wait, 2),
    })

    elapsed = start_wait + SAFETY_GAP_SECONDS
    for next_node in path.nodes[1:]:
        edge_distance = _segment_distance(router.graph, current_node, next_node)
        edge_seconds = edge_distance / speed
        arrival = cursor + timedelta(seconds=edge_seconds)
        wait_seconds = _resolve_node_wait(
            updated_reservations, next_node, arrival, agv.id, task.id
        )
        if wait_seconds > 0:
            waits.append({
                "node_id": next_node,
                "seconds": round(wait_seconds, 2),
                "reason": "避碰等待",
            })
            arrival += timedelta(seconds=wait_seconds)
            elapsed += wait_seconds

        service_seconds = 0.0
        if next_node == task.pickup_node_id and next_node != task.dropoff_node_id:
            service_seconds = LOADING_SECONDS
        if next_node == task.dropoff_node_id:
            service_seconds = max(service_seconds, UNLOADING_SECONDS)

        departure = arrival + timedelta(seconds=service_seconds + SAFETY_GAP_SECONDS)
        _reserve_node(
            updated_reservations, next_node, arrival, departure, agv.id, task.id
        )
        elapsed += edge_seconds + service_seconds
        timeline.append({
            "node_id": next_node,
            "arrival_seconds": round(elapsed, 2),
            "departure_seconds": round(elapsed + SAFETY_GAP_SECONDS, 2),
            "wait_seconds": round(wait_seconds, 2),
        })
        legs.append({
            "from_node": current_node,
            "to_node": next_node,
            "distance": round(edge_distance, 3),
            "seconds": round(edge_seconds + wait_seconds, 2),
        })
        cursor = departure
        current_node = next_node

    route = {
        "nodes": path.nodes,
        "legs": legs,
        "timeline": timeline,
        "waits": waits,
        "pickup_node": task.pickup_node_id,
        "dropoff_node": task.dropoff_node_id,
        "start_node": start_node,
    }
    return route, cursor, updated_reservations, path.distance, sum(
        item["seconds"] for item in waits
    )

def _is_agv_compatible(agv: AGV, task: TransportTask) -> bool:
    return (
        agv.status == AGV.Status.IDLE
        and agv.battery_percent >= MIN_BATTERY_PERCENT
        and agv.payload_capacity >= task.cargo_weight
    )


def evaluate_solution(
    tasks: dict[int, TransportTask],
    agvs: list[AGV],
    orders: dict[int, list[int]],
    router: GraphRouter,
    now: datetime,
    weights: dict[str, float] | None = None,
    allow_partial: bool = False,
) -> ScheduleEvaluation:
    """完整评估任务分配、执行顺序、电池约束和多车时间窗冲突。"""
    weights = weights or {}
    agv_weight = weights.get("agv_weight", DEFAULT_AGV_WEIGHT)
    wait_weight = weights.get("wait_weight", DEFAULT_WAIT_WEIGHT)
    distance_weight = weights.get("distance_weight", DEFAULT_DISTANCE_WEIGHT)
    priority_weight = weights.get("priority_weight", DEFAULT_PRIORITY_WEIGHT)

    agv_map = {agv.id: agv for agv in agvs}
    reservations: dict[int, list[dict]] = {}
    plans: dict[int, TaskPlan] = {}
    agv_summary: dict[int, dict] = {}
    assignments: dict[int, int] = {}
    flattened = []

    for agv_id in sorted(orders):
        sequence = orders.get(agv_id, [])
        if not sequence:
            continue
        agv = agv_map.get(agv_id)
        if agv is None:
            return ScheduleEvaluation(False, message=f"AGV {agv_id} 不存在。")

        current_node = agv.current_node_id
        cursor = now
        agv_distance = 0.0
        agv_wait = 0.0
        agv_plans: list[int] = []
        battery_remaining = agv.battery_percent

        for sequence_index, task_id in enumerate(sequence, start=1):
            task = tasks.get(task_id)
            if task is None:
                return ScheduleEvaluation(False, message=f"任务 {task_id} 不存在。")
            if not _is_agv_compatible(agv, task):
                return ScheduleEvaluation(
                    False, message=f"{agv.code} 状态、电量或载重不满足 {task.task_no}。"
                )
            try:
                route, finish, reservations, distance, wait_seconds = _plan_task_route(
                    agv, task, router, cursor, reservations, current_node
                )
            except SchedulingError as exc:
                return ScheduleEvaluation(False, message=str(exc))

            battery_remaining -= distance * BATTERY_DROP_PER_METER
            if battery_remaining < MIN_BATTERY_PERCENT:
                return ScheduleEvaluation(
                    False, message=f"{agv.code} 执行该任务序列后电量不足。"
                )

            plans[task_id] = TaskPlan(
                task_id=task.id,
                task_no=task.task_no,
                agv_id=agv.id,
                agv_code=agv.code,
                sequence_number=sequence_index,
                start_time=cursor,
                end_time=finish,
                route=route,
                distance=round(distance, 3),
                wait_seconds=round(wait_seconds, 3),
                duration_seconds=round((finish - cursor).total_seconds(), 3),
            )
            assignments[task_id] = agv.id
            flattened.append(task_id)
            agv_plans.append(task_id)
            agv_distance += distance
            agv_wait += wait_seconds
            current_node = task.dropoff_node_id
            cursor = finish

        agv_summary[agv.id] = {
            "agv_id": agv.id,
            "agv_code": agv.code,
            "task_ids": agv_plans,
            "start_time": now,
            "end_time": cursor,
            "distance": round(agv_distance, 3),
            "wait_seconds": round(agv_wait, 3),
            "battery_remaining": round(battery_remaining, 2),
        }

    if len(flattened) != len(set(flattened)):
        return ScheduleEvaluation(False, message="存在重复分配的任务。")
    if not allow_partial and set(flattened) != set(tasks):
        return ScheduleEvaluation(False, message="存在未分配任务。")

    finish_times = [plan.end_time for plan in plans.values()]
    makespan = max((item - now).total_seconds() for item in finish_times)
    total_completion = sum((plan.end_time - now).total_seconds() for plan in plans.values())
    total_distance = sum(plan.distance for plan in plans.values())
    total_wait = sum(plan.wait_seconds for plan in plans.values())
    used_agvs = len(agv_summary)
    weighted_completion = sum(
        (plan.end_time - now).total_seconds() * tasks[task_id].priority
        for task_id, plan in plans.items()
    )
    objective = (
        makespan
        + agv_weight * used_agvs
        + wait_weight * total_wait
        + distance_weight * total_distance
        + priority_weight * weighted_completion
    )
    return ScheduleEvaluation(
        feasible=True,
        objective=round(objective, 4),
        makespan_seconds=round(makespan, 3),
        total_completion_seconds=round(total_completion, 3),
        total_distance=round(total_distance, 3),
        total_wait_seconds=round(total_wait, 3),
        used_agvs=used_agvs,
        assignments=assignments,
        orders={agv_id: list(order) for agv_id, order in orders.items() if order},
        plans=plans,
        agv_summary=agv_summary,
    )

def _first_task_distance(router: GraphRouter, start_node: int, task: TransportTask) -> float:
    return (
        router.path(start_node, task.pickup_node_id).distance
        + router.path(task.pickup_node_id, task.dropoff_node_id).distance
    )


def _transition_distance(router: GraphRouter, first: TransportTask, second: TransportTask) -> float:
    return (
        router.path(first.dropoff_node_id, second.pickup_node_id).distance
        + router.path(second.pickup_node_id, second.dropoff_node_id).distance
    )


def _sequence_distance(
    router: GraphRouter, agv: AGV, tasks: list[TransportTask], order: list[int]
) -> float:
    if not order:
        return 0.0
    task_map = {task.id: task for task in tasks}
    total = _first_task_distance(router, agv.current_node_id, task_map[order[0]])
    for left, right in zip(order, order[1:]):
        total += _transition_distance(router, task_map[left], task_map[right])
    return total


def _two_opt_single_order(
    router: GraphRouter, agv: AGV, tasks: list[TransportTask], order: list[int]
) -> list[int]:
    improved = True
    best_order = list(order)
    while improved:
        improved = False
        for start in range(1, len(best_order) - 1):
            for end in range(start + 1, len(best_order)):
                candidate = (
                    best_order[:start]
                    + list(reversed(best_order[start:end + 1]))
                    + best_order[end + 1:]
                )
                if _sequence_distance(router, agv, tasks, candidate) + 1e-9 < _sequence_distance(
                    router, agv, tasks, best_order
                ):
                    best_order = candidate
                    improved = True
    return best_order

def optimize_single_order(
    agv: AGV, tasks: list[TransportTask], router: GraphRouter
) -> list[int]:
    """单台多任务顺序优化：小规模精确 DP，大规模最近邻 + 2-opt。"""
    if len(tasks) <= 1:
        return [task.id for task in tasks]
    task_map = {task.id: task for task in tasks}
    task_ids = [task.id for task in tasks]

    if len(tasks) <= 10:
        dp: dict[tuple[int, int], tuple[float, tuple[int, ...]]] = {}
        for index, task in enumerate(tasks):
            distance = _first_task_distance(router, agv.current_node_id, task)
            dp[(1 << index, index)] = (distance, (index,))

        for mask in range(1, 1 << len(tasks)):
            for last_index in range(len(tasks)):
                state = (mask, last_index)
                if state not in dp or not (mask & (1 << last_index)):
                    continue
                current_cost, current_path = dp[state]
                for next_index, next_task in enumerate(tasks):
                    if mask & (1 << next_index):
                        continue
                    next_mask = mask | (1 << next_index)
                    next_cost = current_cost + _transition_distance(
                        router, tasks[last_index], next_task
                    )
                    next_state = (next_mask, next_index)
                    existing = dp.get(next_state)
                    if existing is None or next_cost < existing[0] - 1e-9:
                        dp[next_state] = (next_cost, current_path + (next_index,))

        full_mask = (1 << len(tasks)) - 1
        _, best_indices = min(
            (dp[(full_mask, index)] for index in range(len(tasks))),
            key=lambda item: item[0],
        )
        return [task_ids[index] for index in best_indices]

    remaining = list(task_ids)
    order = []
    current_node = agv.current_node_id
    while remaining:
        selected = min(
            remaining,
            key=lambda task_id: _first_task_distance(
                router, current_node, task_map[task_id]
            ),
        )
        order.append(selected)
        current_node = task_map[selected].dropoff_node_id
        remaining.remove(selected)
    return _two_opt_single_order(router, agv, tasks, order)

def _ordered_tasks(tasks: Iterable[TransportTask]) -> list[TransportTask]:
    return sorted(tasks, key=lambda item: (-item.priority, item.created_at, item.id))


def build_baseline_solution(
    tasks: list[TransportTask],
    agvs: list[AGV],
    router: GraphRouter,
    now: datetime,
    weights: dict[str, float] | None = None,
) -> ScheduleEvaluation:
    """基准算法：按优先级顺序选择“最早完成”的 AGV 并追加任务。"""
    orders = {agv.id: [] for agv in agvs}
    task_map = {task.id: task for task in tasks}
    for task in _ordered_tasks(tasks):
        best = None
        for agv in agvs:
            if agv.payload_capacity < task.cargo_weight:
                continue
            candidate = {agv_id: list(order) for agv_id, order in orders.items()}
            candidate[agv.id].append(task.id)
            evaluation = evaluate_solution(
                task_map, agvs, candidate, router, now, weights, allow_partial=True
            )
            if not evaluation.feasible:
                continue
            finish = evaluation.plans[task.id].end_time
            score = finish.timestamp()
            if best is None or score < best["score"]:
                best = {"score": score, "orders": candidate}
        if best is None:
            return ScheduleEvaluation(False, message=f"基准算法无法分配 {task.task_no}。")
        orders = best["orders"]
    return evaluate_solution(task_map, agvs, orders, router, now, weights)

def build_insertion_solution(
    tasks: list[TransportTask],
    agvs: list[AGV],
    router: GraphRouter,
    now: datetime,
    weights: dict[str, float] | None = None,
    task_order: list[TransportTask] | None = None,
) -> ScheduleEvaluation:
    """最优插入初始化：对每个任务尝试所有 AGV、所有插入位置。"""
    orders = {agv.id: [] for agv in agvs}
    task_map = {task.id: task for task in tasks}
    current_evaluation = None

    ordered_tasks = task_order if task_order is not None else _ordered_tasks(tasks)
    for task in ordered_tasks:
        best = None
        for agv in agvs:
            if agv.payload_capacity < task.cargo_weight:
                continue
            base_order = list(orders[agv.id])
            for position in range(len(base_order) + 1):
                candidate = {agv_id: list(order) for agv_id, order in orders.items()}
                candidate[agv.id].insert(position, task.id)
                evaluation = evaluate_solution(
                task_map, agvs, candidate, router, now, weights, allow_partial=True
            )
                if not evaluation.feasible:
                    continue
                if best is None or evaluation.objective < best.objective:
                    best = evaluation

        if best is None:
            return ScheduleEvaluation(False, message=f"没有可行方案分配 {task.task_no}。")
        orders = {agv.id: list(best.orders.get(agv.id, [])) for agv in agvs}
        current_evaluation = best

    return current_evaluation or ScheduleEvaluation(False, message="没有待调度任务。")

def _build_relocate_moves(
    orders: dict[int, list[int]], task_ids: list[int], agv_ids: list[int]
) -> list[dict[int, list[int]]]:
    candidates = []
    for source_agv in agv_ids:
        source_order = orders.get(source_agv, [])
        for source_index, task_id in enumerate(source_order):
            for target_agv in agv_ids:
                if target_agv == source_agv:
                    base_target = list(source_order)
                    base_target.pop(source_index)
                else:
                    base_target = list(orders.get(target_agv, []))
                for position in range(len(base_target) + 1):
                    if target_agv == source_agv and position == source_index:
                        continue
                    candidate = {agv_id: list(orders.get(agv_id, [])) for agv_id in agv_ids}
                    if target_agv == source_agv:
                        candidate[target_agv] = list(base_target)
                        candidate[target_agv].insert(position, task_id)
                    else:
                        candidate[source_agv].remove(task_id)
                        candidate[target_agv].insert(position, task_id)
                    candidates.append(candidate)
    return candidates


def improve_solution_vns(
    initial: ScheduleEvaluation,
    tasks: dict[int, TransportTask],
    agvs: list[AGV],
    router: GraphRouter,
    now: datetime,
    weights: dict[str, float] | None = None,
    max_rounds: int = 50,
    max_neighbors: int = 350,
) -> tuple[ScheduleEvaluation, int]:
    """变邻域局部搜索：不断执行跨 AGV/序列位置迁移，直到无法改进。"""
    current = initial
    iterations = 0
    rng = random.Random(20260928)
    task_ids = list(tasks)
    agv_ids = [agv.id for agv in agvs]

    for _round in range(max_rounds):
        moves = _build_relocate_moves(current.orders, task_ids, agv_ids)
        if not moves:
            break
        if len(moves) > max_neighbors:
            moves = rng.sample(moves, max_neighbors)
        improved = False
        for candidate in moves:
            iterations += 1
            evaluation = evaluate_solution(tasks, agvs, candidate, router, now, weights)
            if evaluation.feasible and evaluation.objective < current.objective - 1e-7:
                current = evaluation
                improved = True
                break
        if not improved:
            break
    return current, iterations

def optimize_single_solution(
    tasks: list[TransportTask],
    agvs: list[AGV],
    router: GraphRouter,
    now: datetime,
    weights: dict[str, float] | None = None,
) -> tuple[ScheduleEvaluation, ScheduleEvaluation, int]:
    """同时选择单台 AGV 并优化该车的多任务执行顺序。"""
    task_map = {task.id: task for task in tasks}
    weights = dict(weights or {})
    weights["agv_weight"] = 0
    best = None
    best_baseline = ScheduleEvaluation(False)
    iterations = 0

    for agv in agvs:
        if any(not _is_agv_compatible(agv, task) for task in tasks):
            continue
        optimized_order = optimize_single_order(agv, tasks, router)
        evaluation = evaluate_solution(
            task_map, [agv], {agv.id: optimized_order}, router, now, weights
        )
        if not evaluation.feasible:
            continue
        selection_score = evaluation.objective + (100 - agv.battery_percent) * 0.25
        iterations += max(1, len(tasks) * len(tasks))
        if best is None or selection_score < best["score"]:
            baseline_order = [task.id for task in tasks]
            baseline = evaluate_solution(
                task_map, [agv], {agv.id: baseline_order}, router, now, weights
            )
            best = {"score": selection_score, "evaluation": evaluation}
            best_baseline = baseline

    if best is None:
        return ScheduleEvaluation(False), ScheduleEvaluation(False), iterations
    return best["evaluation"], best_baseline, iterations


def optimize_multi_solution(
    tasks: list[TransportTask],
    agvs: list[AGV],
    router: GraphRouter,
    now: datetime,
    weights: dict[str, float] | None = None,
) -> tuple[ScheduleEvaluation, ScheduleEvaluation, int]:
    """最优插入初始化 + 变邻域搜索的多 AGV 联合优化。"""
    task_map = {task.id: task for task in tasks}
    baseline = build_baseline_solution(tasks, agvs, router, now, weights)
    initial = build_insertion_solution(tasks, agvs, router, now, weights)
    initialization_evaluations = 0
    if len(tasks) <= 8:
        rng = random.Random(20260928)
        for _ in range(60):
            shuffled = list(tasks)
            rng.shuffle(shuffled)
            candidate = build_insertion_solution(
                tasks, agvs, router, now, weights, task_order=shuffled
            )
            initialization_evaluations += max(1, len(tasks) * len(tasks) * 2)
            if candidate.feasible and candidate.objective < initial.objective:
                initial = candidate
    if not initial.feasible:
        return ScheduleEvaluation(False, message=initial.message), baseline, 0
    optimized, iterations = improve_solution_vns(
        initial, task_map, agvs, router, now, weights
    )
    iterations += initialization_evaluations
    if baseline.feasible and baseline.objective < optimized.objective:
        return baseline, baseline, iterations
    return optimized, baseline, iterations

def _evaluation_payload(evaluation: ScheduleEvaluation) -> dict:
    if not evaluation.feasible:
        return {"feasible": False, "message": evaluation.message}
    return {
        "feasible": True,
        "objective": evaluation.objective,
        "makespan_seconds": evaluation.makespan_seconds,
        "total_completion_seconds": evaluation.total_completion_seconds,
        "total_distance": evaluation.total_distance,
        "total_wait_seconds": evaluation.total_wait_seconds,
        "used_agvs": evaluation.used_agvs,
        "agv_summary": [
            {
                **summary,
                "start_time": summary["start_time"].isoformat(),
                "end_time": summary["end_time"].isoformat(),
            }
            for summary in evaluation.agv_summary.values()
        ],
        "tasks": [
            {
                "task_id": plan.task_id,
                "task_no": plan.task_no,
                "agv_id": plan.agv_id,
                "agv_code": plan.agv_code,
                "sequence_number": plan.sequence_number,
                "start_time": plan.start_time.isoformat(),
                "end_time": plan.end_time.isoformat(),
                "duration_seconds": plan.duration_seconds,
                "distance": plan.distance,
                "wait_seconds": plan.wait_seconds,
            }
            for plan in evaluation.plans.values()
        ],
    }


def _persist_schedule(
    evaluation: ScheduleEvaluation,
    baseline: ScheduleEvaluation,
    tasks: list[TransportTask],
    algorithm: str,
    objective_name: str,
    iteration_count: int,
) -> ScheduleRun:
    improvement = 0.0
    baseline_makespan = baseline.makespan_seconds if baseline.feasible else 0.0
    if baseline.feasible and baseline.objective > 0:
        improvement = round(
            max(0.0, (baseline.objective - evaluation.objective) / baseline.objective * 100),
            2,
        )

    assignments = {
        str(summary["agv_id"]): {
            "agv_code": summary["agv_code"],
            "task_ids": summary["task_ids"],
            "end_time": summary["end_time"].isoformat(),
            "distance": summary["distance"],
            "wait_seconds": summary["wait_seconds"],
            "battery_remaining": summary["battery_remaining"],
        }
        for summary in evaluation.agv_summary.values()
    }
    run = ScheduleRun.objects.create(
        algorithm=algorithm,
        objective_name=objective_name,
        task_count=len(tasks),
        agv_count=evaluation.used_agvs,
        makespan_seconds=evaluation.makespan_seconds,
        total_distance=evaluation.total_distance,
        total_wait_seconds=evaluation.total_wait_seconds,
        iteration_count=iteration_count,
        baseline_makespan_seconds=baseline_makespan,
        improvement_percent=improvement,
        assignments=assignments,
        metrics={
            "objective": evaluation.objective,
            "total_completion_seconds": evaluation.total_completion_seconds,
            "baseline_objective": baseline.objective if baseline.feasible else None,
            "baseline_makespan_seconds": baseline_makespan if baseline.feasible else None,
            "algorithm": algorithm,
        },
    )

    active_agv_ids = set()
    for task in tasks:
        plan = evaluation.plans[task.id]
        task.assigned_agv_id = plan.agv_id
        task.schedule_run = run
        task.sequence_number = plan.sequence_number
        task.planned_wait_seconds = plan.wait_seconds
        task.optimization_details = {
            "objective": evaluation.objective,
            "makespan_seconds": evaluation.makespan_seconds,
            "baseline_makespan_seconds": baseline_makespan,
            "improvement_percent": improvement,
            "algorithm": algorithm,
        }
        task.status = TransportTask.Status.ASSIGNED
        task.route = plan.route
        task.total_distance = plan.distance
        task.estimated_duration = plan.duration_seconds
        task.scheduled_start = plan.start_time
        task.scheduled_end = plan.end_time
        task.started_at = None
        task.completed_at = None
        task.notes = (
            f"{run.get_algorithm_display()}：{plan.agv_code} 第 {plan.sequence_number} 项"
            + (f"，避碰等待 {plan.wait_seconds:.1f}s" if plan.wait_seconds > 0 else "")
        )
        task.save(
            update_fields=[
                "assigned_agv", "schedule_run", "sequence_number", "planned_wait_seconds",
                "optimization_details", "status", "route", "total_distance",
                "estimated_duration", "scheduled_start", "scheduled_end", "started_at",
                "completed_at", "notes", "updated_at",
            ]
        )
        DispatchRecord.objects.create(
            task=task,
            agv_id=plan.agv_id,
            algorithm=(
                DispatchRecord.Algorithm.SINGLE
                if algorithm == ScheduleRun.Algorithm.SINGLE_SEQUENCE
                else DispatchRecord.Algorithm.BATCH
            ),
            success=True,
            message=task.notes,
            route=plan.route,
            distance=plan.distance,
            estimated_duration=plan.duration_seconds,
        )
        active_agv_ids.add(plan.agv_id)

    AGV.objects.filter(id__in=active_agv_ids).update(status=AGV.Status.BUSY)
    return run

def dispatch_single_sequence(
    task_ids: list[int],
    agv_id: int | None = None,
    weights: dict[str, float] | None = None,
) -> dict:
    """单台 AGV 多任务调度：选择 AGV 并优化所有任务执行顺序。"""
    if not task_ids:
        raise SchedulingError("至少选择一个任务。")
    if len(set(task_ids)) != len(task_ids):
        raise SchedulingError("任务列表中不能包含重复任务。")
    with transaction.atomic():
        tasks_by_id = {
            task.id: task
            for task in TransportTask.objects.select_for_update()
            .select_related("pickup_node", "dropoff_node")
            .filter(id__in=task_ids)
        }
        missing = [task_id for task_id in task_ids if task_id not in tasks_by_id]
        if missing:
            raise SchedulingError(f"任务不存在：{missing}")
        tasks = [tasks_by_id[task_id] for task_id in task_ids]
        invalid = [task.task_no for task in tasks if task.status != TransportTask.Status.PENDING]
        if invalid:
            raise SchedulingError(f"仅待调度任务可参与优化：{', '.join(invalid)}")

        agvs_query = AGV.objects.select_for_update().select_related("current_node").filter(
            status=AGV.Status.IDLE
        )
        if agv_id is not None:
            agvs_query = agvs_query.filter(id=agv_id)
        agvs = list(agvs_query.order_by("code"))
        if not agvs:
            raise SchedulingError("没有可用的空闲 AGV。")

        now = timezone.now()
        router = GraphRouter(build_graph())
        evaluation, baseline, iterations = optimize_single_solution(
            tasks, agvs, router, now, weights
        )
        if not evaluation.feasible:
            raise SchedulingError(evaluation.message or "单台 AGV 顺序优化失败。")
        run = _persist_schedule(
            evaluation=evaluation,
            baseline=baseline,
            tasks=tasks,
            algorithm=ScheduleRun.Algorithm.SINGLE_SEQUENCE,
            objective_name="最小化总搬运时间与路径长度",
            iteration_count=iterations,
        )
        agv_code = evaluation.plans[tasks[0].id].agv_code
        message = (
            f"单台 AGV 顺序优化完成：{agv_code} 执行 {len(tasks)} 个任务，"
            f"完工时间 {evaluation.makespan_seconds:.1f}s，"
            f"相比输入顺序改进 {run.improvement_percent:.1f}%"
        )
        return {
            "message": message,
            "algorithm": run.algorithm,
            "run": run,
            "tasks": tasks,
            "evaluation": _evaluation_payload(evaluation),
            "baseline": _evaluation_payload(baseline),
        }


def dispatch_single(task_id: int, agv_id: int | None = None) -> TransportTask:
    """兼容旧接口：单个任务的单台 AGV 调度。"""
    result = dispatch_single_sequence([task_id], agv_id)
    return result["tasks"][0]

def dispatch_batch(
    task_ids: list[int] | None = None,
    limit: int = 30,
    weights: dict[str, float] | None = None,
) -> dict:
    """多台 AGV 联合优化：任务分配、车辆内顺序和避碰时间窗统一求解。"""
    if task_ids and len(set(task_ids)) != len(task_ids):
        raise SchedulingError("任务列表中不能包含重复任务。")
    with transaction.atomic():
        query = TransportTask.objects.select_for_update().select_related(
            "pickup_node", "dropoff_node"
        ).filter(status=TransportTask.Status.PENDING)
        if task_ids:
            query = query.filter(id__in=task_ids)
        tasks = list(query.order_by("-priority", "created_at")[:limit])
        if not tasks:
            raise SchedulingError("没有可参与批量调度的待处理任务。")

        agvs = list(
            AGV.objects.select_for_update()
            .select_related("current_node")
            .filter(status=AGV.Status.IDLE)
            .order_by("code")
        )
        if not agvs:
            raise SchedulingError("当前没有空闲 AGV。")

        now = timezone.now()
        router = GraphRouter(build_graph())
        evaluation, baseline, iterations = optimize_multi_solution(
            tasks, agvs, router, now, weights
        )
        if not evaluation.feasible:
            raise SchedulingError(evaluation.message or "多台 AGV 联合优化失败。")

        run = _persist_schedule(
            evaluation=evaluation,
            baseline=baseline,
            tasks=tasks,
            algorithm=ScheduleRun.Algorithm.MULTI_VNS,
            objective_name="最小化完工时间、AGV 数量、优先级加权完成时间与避碰等待",
            iteration_count=iterations,
        )
        message = (
            f"多台 AGV 联合优化完成：{len(tasks)} 个任务，"
            f"使用 {evaluation.used_agvs} 台 AGV，"
            f"makespan {evaluation.makespan_seconds:.1f}s，"
            f"相比基准算法改进 {run.improvement_percent:.1f}%"
        )
        return {
            "message": message,
            "algorithm": run.algorithm,
            "run": run,
            "tasks": tasks,
            "evaluation": _evaluation_payload(evaluation),
            "baseline": _evaluation_payload(baseline),
        }

def start_task(task_id: int) -> TransportTask:
    with transaction.atomic():
        task = (
            TransportTask.objects.select_for_update()
            .select_related("assigned_agv")
            .get(pk=task_id)
        )
        if task.status != TransportTask.Status.ASSIGNED or not task.assigned_agv:
            raise SchedulingError("只有已分配任务可以开始执行。")
        if task.schedule_run_id:
            previous_pending = TransportTask.objects.filter(
                schedule_run_id=task.schedule_run_id,
                assigned_agv_id=task.assigned_agv_id,
                sequence_number__lt=task.sequence_number,
            ).exclude(status__in=[TransportTask.Status.COMPLETED, TransportTask.Status.CANCELLED])
            if previous_pending.exists():
                raise SchedulingError("存在未完成的前序任务，请按优化顺序执行。")
        task.status = TransportTask.Status.IN_PROGRESS
        task.started_at = timezone.now()
        task.save(update_fields=["status", "started_at", "updated_at"])
        AGV.objects.filter(pk=task.assigned_agv_id).update(status=AGV.Status.BUSY)
    return task


def complete_task(task_id: int) -> TransportTask:
    with transaction.atomic():
        task = (
            TransportTask.objects.select_for_update()
            .select_related("assigned_agv")
            .get(pk=task_id)
        )
        if task.status != TransportTask.Status.IN_PROGRESS or not task.assigned_agv:
            raise SchedulingError("只有执行中的任务可以完成。")

        agv = AGV.objects.select_for_update().get(pk=task.assigned_agv_id)
        agv.current_node = task.dropoff_node
        agv.battery_percent = max(
            0.0, agv.battery_percent - task.total_distance * BATTERY_DROP_PER_METER
        )
        has_next = TransportTask.objects.filter(
            assigned_agv=agv,
            status=TransportTask.Status.ASSIGNED,
        ).exclude(pk=task.id).exists()
        agv.status = AGV.Status.BUSY if has_next else AGV.Status.IDLE
        agv.save(
            update_fields=["current_node", "battery_percent", "status", "last_heartbeat"]
        )

        task.status = TransportTask.Status.COMPLETED
        task.completed_at = timezone.now()
        task.save(update_fields=["status", "completed_at", "updated_at"])
    return task


def cancel_task(task_id: int) -> TransportTask:
    with transaction.atomic():
        task = (
            TransportTask.objects.select_for_update()
            .select_related("assigned_agv")
            .get(pk=task_id)
        )
        if task.status in [TransportTask.Status.COMPLETED, TransportTask.Status.IN_PROGRESS]:
            raise SchedulingError("已完成或执行中的任务不能取消。")
        old_agv_id = task.assigned_agv_id
        task.status = TransportTask.Status.CANCELLED
        task.save(update_fields=["status", "updated_at"])
        if old_agv_id:
            has_more = TransportTask.objects.filter(
                assigned_agv_id=old_agv_id,
                status__in=[
                    TransportTask.Status.ASSIGNED,
                    TransportTask.Status.IN_PROGRESS,
                ],
            ).exists()
            if not has_more:
                AGV.objects.filter(pk=old_agv_id).update(status=AGV.Status.IDLE)
    return task
