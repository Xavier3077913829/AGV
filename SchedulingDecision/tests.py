from types import SimpleNamespace
from unittest import mock

from django.core.management import call_command
from django.test import SimpleTestCase, TestCase
from rest_framework.test import APIClient

from .models import AGV, DispatchRecord, MapNode, ScheduleRun, TransportTask
from .services import (
    SchedulingError,
    _build_or_opt_moves,
    _build_relocate_moves,
    _build_swap_moves,
    _build_two_opt_moves,
    complete_task,
    dispatch_batch,
    dispatch_single,
    dispatch_single_sequence,
    improve_solution_vns,
    start_task,
)


class SchedulingFlowTests(TestCase):
    @classmethod
    def setUpTestData(cls):
        call_command("seed_demo", "--reset", verbosity=0)

    def test_single_dispatch_builds_route_and_updates_status(self):
        task = TransportTask.objects.filter(status=TransportTask.Status.PENDING).first()
        result = dispatch_single(task.id)
        result.refresh_from_db()
        self.assertEqual(result.status, TransportTask.Status.ASSIGNED)
        self.assertIsNotNone(result.assigned_agv)
        self.assertGreater(result.total_distance, 0)
        self.assertIn(result.pickup_node_id, result.route["nodes"])
        self.assertIn(result.dropoff_node_id, result.route["nodes"])
        self.assertEqual(result.assigned_agv.status, AGV.Status.BUSY)

    def test_single_agv_multi_task_sequence_is_optimized(self):
        task_ids = list(
            TransportTask.objects.filter(status=TransportTask.Status.PENDING)
            .order_by("created_at")
            .values_list("id", flat=True)[:5]
        )
        result = dispatch_single_sequence(task_ids)
        tasks = list(
            TransportTask.objects.filter(id__in=task_ids)
            .select_related("assigned_agv", "schedule_run")
            .order_by("sequence_number")
        )
        self.assertEqual(len(tasks), 5)
        self.assertEqual(len({task.assigned_agv_id for task in tasks}), 1)
        self.assertEqual([task.sequence_number for task in tasks], [1, 2, 3, 4, 5])
        self.assertTrue(all(task.schedule_run_id for task in tasks))
        self.assertEqual(
            result["run"].algorithm, ScheduleRun.Algorithm.SINGLE_SEQUENCE
        )
        self.assertGreaterEqual(result["run"].improvement_percent, 0)

    def test_batch_dispatch_optimizes_multiple_agvs(self):
        result = dispatch_batch()
        evaluation = result["evaluation"]
        self.assertEqual(len(result["tasks"]), 6)
        self.assertGreaterEqual(evaluation["used_agvs"], 2)
        self.assertLessEqual(evaluation["used_agvs"], 4)
        self.assertGreater(evaluation["makespan_seconds"], 0)
        self.assertEqual(
            result["run"].algorithm, ScheduleRun.Algorithm.MULTI_VNS
        )
        self.assertEqual(
            TransportTask.objects.filter(status=TransportTask.Status.ASSIGNED).count(), 6
        )
        self.assertEqual(DispatchRecord.objects.filter(algorithm="batch").count(), 6)

    def test_optimized_sequence_must_be_executed_in_order(self):
        task_ids = list(
            TransportTask.objects.filter(status=TransportTask.Status.PENDING)
            .values_list("id", flat=True)[:3]
        )
        dispatch_single_sequence(task_ids)
        ordered = list(
            TransportTask.objects.filter(id__in=task_ids).order_by("sequence_number")
        )
        with self.assertRaises(SchedulingError):
            start_task(ordered[1].id)
        start_task(ordered[0].id)
        complete_task(ordered[0].id)
        start_task(ordered[1].id)
        complete_task(ordered[1].id)
        start_task(ordered[2].id)
        complete_task(ordered[2].id)
        self.assertEqual(
            TransportTask.objects.filter(id__in=task_ids, status="completed").count(), 3
        )
    def test_task_lifecycle_updates_agv_position_and_battery(self):
        task = TransportTask.objects.filter(status=TransportTask.Status.PENDING).first()
        dispatch_single(task.id)
        task.refresh_from_db()
        agv = task.assigned_agv
        original_battery = agv.battery_percent
        start_task(task.id)
        complete_task(task.id)
        task.refresh_from_db()
        agv.refresh_from_db()
        self.assertEqual(task.status, TransportTask.Status.COMPLETED)
        self.assertEqual(agv.current_node_id, task.dropoff_node_id)
        self.assertLess(agv.battery_percent, original_battery)


class APITests(TestCase):
    @classmethod
    def setUpTestData(cls):
        call_command("seed_demo", "--reset", verbosity=0)

    def setUp(self):
        self.client = APIClient()
        self.host = {"HTTP_HOST": "localhost"}

    def test_overview_and_map_endpoints(self):
        overview = self.client.get("/api/overview/", **self.host)
        graph = self.client.get("/api/map/", **self.host)
        self.assertEqual(overview.status_code, 200)
        self.assertEqual(overview.data["agv_total"], 4)
        self.assertEqual(graph.status_code, 200)
        self.assertEqual(len(graph.data["nodes"]), 76)
        self.assertEqual(len(graph.data["edges"]), 103)
        self.assertEqual(len(graph.data["obstacles"]), 10)

    def test_single_schedule_endpoint_accepts_multiple_tasks(self):
        task_ids = list(
            TransportTask.objects.filter(status=TransportTask.Status.PENDING)
            .values_list("id", flat=True)[:4]
        )
        response = self.client.post(
            "/api/schedules/single/",
            {"task_ids": task_ids},
            format="json",
            **self.host,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data["tasks"]), 4)
        self.assertEqual(response.data["run"]["algorithm"], "single_sequence")
        self.assertGreater(response.data["evaluation"]["makespan_seconds"], 0)

    def test_path_planning_endpoint_avoids_obstacles(self):
        start = MapNode.objects.get(code="S2")
        end = MapNode.objects.get(code="S6")
        response = self.client.post(
            "/api/path-planning/",
            {"start_node_id": start.id, "end_node_id": end.id},
            format="json",
            **self.host,
        )
        self.assertEqual(response.status_code, 200)
        self.assertGreater(response.data["distance"], 0)
        self.assertEqual(response.data["route"]["nodes"][0], start.id)
        self.assertEqual(response.data["route"]["nodes"][-1], end.id)

    def test_batch_schedule_endpoint_returns_optimization_metrics(self):
        response = self.client.post(
            "/api/schedules/batch/", {}, format="json", **self.host
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["run"]["task_count"], 6)
        self.assertGreaterEqual(response.data["run"]["agv_count"], 2)
        self.assertTrue(response.data["evaluation"]["feasible"])


class NeighborhoodTests(SimpleTestCase):
    """变邻域算子与变邻域下降的纯逻辑测试（不需要数据库）。"""

    AGV_IDS = [1, 2, 3]
    TASK_IDS = [101, 102, 103, 104, 105, 106]
    INITIAL_ORDERS = {1: [101, 102, 103], 2: [104], 3: [105, 106]}

    def _assert_task_multiset_preserved(self, candidate):
        """每个任务必须出现且只出现一次（所有邻域都不能丢任务或复制任务）。"""
        flattened = [task_id for order in candidate.values() for task_id in order]
        self.assertEqual(len(flattened), len(self.TASK_IDS))
        self.assertEqual(sorted(flattened), sorted(self.TASK_IDS))

    def test_relocate_moves_preserve_tasks(self):
        moves = _build_relocate_moves(self.INITIAL_ORDERS, self.TASK_IDS, self.AGV_IDS)
        self.assertGreater(len(moves), 0)
        for candidate in moves:
            self._assert_task_multiset_preserved(candidate)

    def test_swap_moves_preserve_tasks(self):
        moves = _build_swap_moves(self.INITIAL_ORDERS, self.AGV_IDS)
        self.assertGreater(len(moves), 0)
        for candidate in moves:
            self._assert_task_multiset_preserved(candidate)

    def test_or_opt_moves_preserve_tasks(self):
        moves = _build_or_opt_moves(self.INITIAL_ORDERS, self.AGV_IDS)
        self.assertGreater(len(moves), 0)
        for candidate in moves:
            self._assert_task_multiset_preserved(candidate)

    def test_two_opt_moves_preserve_tasks(self):
        moves = _build_two_opt_moves(self.INITIAL_ORDERS, self.AGV_IDS)
        self.assertGreater(len(moves), 0)
        for candidate in moves:
            self._assert_task_multiset_preserved(candidate)

    def test_vns_reduces_objective_and_is_reproducible(self):
        """构造一个距离矩阵，验证 VND 能下降目标值，并且同种子结果一致。"""

        def cost(orders):
            total = 0.0
            for agv_id, order in orders.items():
                if not order:
                    continue
                total += 2.0 * agv_id + 3.0 * order[0]  # 起点成本
                for first, second in zip(order, order[1:]):
                    total += abs(first - second) * 1.5  # 任务间转移成本
            return total

        def fake_evaluate(tasks, agvs, orders, router, now, weights=None, **kwargs):
            normalized = {int(key): list(value) for key, value in orders.items()}
            flattened = [task_id for order in normalized.values() for task_id in order]
            feasible = sorted(flattened) == sorted(self.TASK_IDS)
            return SimpleNamespace(
                feasible=feasible,
                objective=cost(normalized) if feasible else float("inf"),
                orders=normalized,
            )

        initial_orders = {agv_id: list(order) for agv_id, order in self.INITIAL_ORDERS.items()}
        initial = SimpleNamespace(
            feasible=True, objective=cost(initial_orders), orders=initial_orders
        )
        agvs = [SimpleNamespace(id=agv_id) for agv_id in self.AGV_IDS]
        tasks = {task_id: None for task_id in self.TASK_IDS}

        with mock.patch("SchedulingDecision.services.evaluate_solution", fake_evaluate):
            first, iterations = improve_solution_vns(initial, tasks, agvs, None, None)
            second, _ = improve_solution_vns(
                SimpleNamespace(
                    feasible=True,
                    objective=cost({k: list(v) for k, v in self.INITIAL_ORDERS.items()}),
                    orders={k: list(v) for k, v in self.INITIAL_ORDERS.items()},
                ),
                tasks,
                agvs,
                None,
                None,
            )

        self.assertTrue(first.feasible)
        self.assertLess(first.objective, initial.objective)  # 目标值必须下降
        self.assertGreater(iterations, 0)  # 迭代次数被记录
        self.assertEqual(first.orders, second.orders)  # 固定种子 → 结果可复现
        self._assert_task_multiset_preserved(first.orders)
