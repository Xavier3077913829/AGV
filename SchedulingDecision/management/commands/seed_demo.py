from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.db import transaction

from SchedulingDecision.models import AGV, DispatchRecord, MapEdge, MapNode, ScheduleRun, TransportTask


NODES = [
    ("P1", "一号停车位", "parking", 0, 0),
    ("S1", "取货口 A", "pickup", 4, 0),
    ("J1", "路口 J1", "intersection", 8, 0),
    ("D1", "放货口 A", "dropoff", 12, 0),
    ("P2", "二号停车位", "parking", 16, 0),
    ("P3", "三号停车位", "parking", 0, 4),
    ("J2", "路口 J2", "intersection", 4, 4),
    ("J3", "路口 J3", "intersection", 8, 4),
    ("J4", "路口 J4", "intersection", 12, 4),
    ("P4", "四号停车位", "parking", 16, 4),
    ("S2", "取货口 B", "pickup", 0, 8),
    ("J5", "路口 J5", "intersection", 4, 8),
    ("D2", "放货口 B", "dropoff", 8, 8),
    ("J6", "路口 J6", "intersection", 12, 8),
    ("P5", "五号停车位", "parking", 16, 8),
    ("P6", "六号停车位", "parking", 0, 12),
    ("D3", "放货口 C", "dropoff", 4, 12),
    ("J7", "路口 J7", "intersection", 8, 12),
    ("S3", "取货口 C", "pickup", 12, 12),
    ("CHG1", "充电站", "charging", 16, 12),
]

EDGE_CODES = [
    ("P1", "S1"), ("S1", "J1"), ("J1", "D1"), ("D1", "P2"),
    ("P3", "J2"), ("J2", "J3"), ("J3", "J4"), ("J4", "P4"),
    ("S2", "J5"), ("J5", "D2"), ("D2", "J6"), ("J6", "P5"),
    ("P6", "D3"), ("D3", "J7"), ("J7", "S3"), ("S3", "CHG1"),
    ("P1", "P3"), ("S1", "J2"), ("J1", "J3"), ("D1", "J4"), ("P2", "P4"),
    ("P3", "S2"), ("J2", "J5"), ("J3", "D2"), ("J4", "J6"), ("P4", "P5"),
    ("S2", "P6"), ("J5", "D3"), ("D2", "J7"), ("J6", "S3"), ("P5", "CHG1"),
]

class Command(BaseCommand):
    help = "初始化 AGV 调度演示数据（地图、AGV、任务和管理员）"

    def add_arguments(self, parser):
        parser.add_argument("--reset", action="store_true", help="清空业务数据后重新初始化")

    @transaction.atomic
    def handle(self, *args, **options):
        if options["reset"]:
            DispatchRecord.objects.all().delete()
            TransportTask.objects.all().delete()
            ScheduleRun.objects.all().delete()
            AGV.objects.all().delete()
            MapEdge.objects.all().delete()
            MapNode.objects.all().delete()

        node_map = {}
        for code, name, node_type, x, y in NODES:
            node, _ = MapNode.objects.update_or_create(
                code=code,
                defaults={"name": name, "node_type": node_type, "x": x, "y": y, "is_active": True},
            )
            node_map[code] = node

        for start_code, end_code in EDGE_CODES:
            start = node_map[start_code]
            end = node_map[end_code]
            distance = round(((start.x - end.x) ** 2 + (start.y - end.y) ** 2) ** 0.5, 2)
            MapEdge.objects.update_or_create(
                start_node=start,
                end_node=end,
                defaults={"distance": distance, "speed_limit": 1.2, "bidirectional": True, "is_active": True},
            )

        agv_specs = [
            ("AGV-01", "天马一号", "T1-100", "P1", 96, 120, 1.2),
            ("AGV-02", "天马二号", "T1-100", "P2", 88, 120, 1.1),
            ("AGV-03", "天马三号", "T2-150", "P5", 73, 150, 1.4),
            ("AGV-04", "天马四号", "T2-150", "P6", 64, 150, 1.35),
        ]
        for code, name, model_name, node_code, battery, capacity, speed in agv_specs:
            AGV.objects.update_or_create(
                code=code,
                defaults={
                    "name": name,
                    "model_name": model_name,
                    "current_node": node_map[node_code],
                    "battery_percent": battery,
                    "payload_capacity": capacity,
                    "speed": speed,
                    "status": AGV.Status.IDLE,
                },
            )

        task_specs = [
            ("TASK-DEMO-001", "S1", "D2", "电子元件箱", 35, 3),
            ("TASK-DEMO-002", "S2", "D3", "汽车零部件", 70, 4),
            ("TASK-DEMO-003", "S3", "D1", "成品包装箱", 45, 2),
            ("TASK-DEMO-004", "S1", "D3", "原料托盘", 95, 3),
            ("TASK-DEMO-005", "S2", "D1", "质检样品", 18, 1),
            ("TASK-DEMO-006", "S3", "D2", "装配工装", 60, 2),
        ]
        for task_no, pickup, dropoff, cargo, weight, priority in task_specs:
            TransportTask.objects.update_or_create(
                task_no=task_no,
                defaults={
                    "pickup_node": node_map[pickup],
                    "dropoff_node": node_map[dropoff],
                    "cargo_name": cargo,
                    "cargo_weight": weight,
                    "priority": priority,
                    "status": TransportTask.Status.PENDING,
                    "assigned_agv": None,
                    "route": {},
                    "total_distance": 0,
                    "estimated_duration": 0,
                    "scheduled_start": None,
                    "scheduled_end": None,
                    "started_at": None,
                    "completed_at": None,
                    "notes": "",
                },
            )

        User = get_user_model()
        if not User.objects.filter(username="admin").exists():
            User.objects.create_superuser("admin", "admin@example.com", "Admin123!")
            self.stdout.write(self.style.SUCCESS("已创建管理员 admin / Admin123!"))
        else:
            self.stdout.write("管理员 admin 已存在，未修改密码。")

        self.stdout.write(self.style.SUCCESS(
            f"演示数据初始化完成：节点 {MapNode.objects.count()}，道路 {MapEdge.objects.count()}，"
            f"AGV {AGV.objects.count()}，待调度任务 {TransportTask.objects.filter(status='pending').count()}"
        ))
