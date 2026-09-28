from django.core.management import call_command
from django.test import TestCase
from rest_framework.test import APIClient

from .models import AGV, DispatchRecord, MapNode, ScheduleRun, TransportTask
from .services import (
    SchedulingError,
    complete_task,
    dispatch_batch,
    dispatch_single,
    dispatch_single_sequence,
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
