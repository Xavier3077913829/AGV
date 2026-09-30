import uuid

from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models


class MapNode(models.Model):
    """仓库地图节点。"""

    class NodeType(models.TextChoices):
        PARKING = "parking", "停车点"
        PICKUP = "pickup", "取货点"
        DROPOFF = "dropoff", "放货点"
        CHARGING = "charging", "充电点"
        INTERSECTION = "intersection", "路径节点"

    code = models.CharField("节点编码", max_length=32, unique=True)
    name = models.CharField("节点名称", max_length=64)
    node_type = models.CharField(
        "节点类型", max_length=20, choices=NodeType.choices, default=NodeType.INTERSECTION
    )
    x = models.FloatField("X 坐标")
    y = models.FloatField("Y 坐标")
    is_active = models.BooleanField("启用", default=True)
    created_at = models.DateTimeField("创建时间", auto_now_add=True)

    class Meta:
        ordering = ["code"]
        verbose_name = "地图节点"
        verbose_name_plural = "地图节点"

    def __str__(self):
        return f"{self.code} - {self.name}"


class MapEdge(models.Model):
    """地图中两个节点之间的可通行道路。"""

    start_node = models.ForeignKey(
        MapNode, on_delete=models.CASCADE, related_name="outgoing_edges", verbose_name="起点"
    )
    end_node = models.ForeignKey(
        MapNode, on_delete=models.CASCADE, related_name="incoming_edges", verbose_name="终点"
    )
    distance = models.FloatField("距离(米)", validators=[MinValueValidator(0.1)])
    speed_limit = models.FloatField(
        "限速(米/秒)", default=1.2, validators=[MinValueValidator(0.1)]
    )
    bidirectional = models.BooleanField("双向通行", default=True)
    is_active = models.BooleanField("启用", default=True)
    created_at = models.DateTimeField("创建时间", auto_now_add=True)

    class Meta:
        ordering = ["start_node__code", "end_node__code"]
        constraints = [
            models.UniqueConstraint(fields=["start_node", "end_node"], name="uniq_edge_pair")
        ]
        verbose_name = "地图道路"
        verbose_name_plural = "地图道路"

    def __str__(self):
        return f"{self.start_node.code} -> {self.end_node.code} ({self.distance}m)"

class MapObstacle(models.Model):
    """地图中的不可通行障碍区域。"""

    code = models.CharField("障碍物编码", max_length=32, unique=True)
    name = models.CharField("障碍物名称", max_length=64, blank=True)
    x = models.FloatField("X 坐标")
    y = models.FloatField("Y 坐标")
    width = models.FloatField("宽度", default=1)
    height = models.FloatField("高度", default=1)
    is_active = models.BooleanField("启用", default=True)
    created_at = models.DateTimeField("创建时间", auto_now_add=True)

    class Meta:
        ordering = ["code"]
        verbose_name = "地图障碍物"
        verbose_name_plural = "地图障碍物"

    def __str__(self):
        return f"{self.code} ({self.x}, {self.y})"

class AGV(models.Model):
    class Status(models.TextChoices):
        IDLE = "idle", "空闲"
        BUSY = "busy", "执行任务"
        CHARGING = "charging", "充电中"
        OFFLINE = "offline", "离线"
        ERROR = "error", "故障"

    code = models.CharField("AGV 编号", max_length=32, unique=True)
    name = models.CharField("名称", max_length=64)
    model_name = models.CharField("型号", max_length=64, blank=True)
    status = models.CharField("状态", max_length=16, choices=Status.choices, default=Status.IDLE)
    current_node = models.ForeignKey(
        MapNode,
        on_delete=models.PROTECT,
        related_name="agvs",
        verbose_name="当前位置",
    )
    battery_percent = models.FloatField(
        "电量(%)", default=100, validators=[MinValueValidator(0), MaxValueValidator(100)]
    )
    payload_capacity = models.FloatField(
        "额定载重(kg)", default=100, validators=[MinValueValidator(0.1)]
    )
    speed = models.FloatField("速度(米/秒)", default=1.0, validators=[MinValueValidator(0.1)])
    last_heartbeat = models.DateTimeField("最后心跳", auto_now=True)
    created_at = models.DateTimeField("创建时间", auto_now_add=True)

    class Meta:
        ordering = ["code"]
        verbose_name = "AGV"
        verbose_name_plural = "AGV"

    def __str__(self):
        return f"{self.code} - {self.name}"

class ScheduleRun(models.Model):
    """一次调度优化的可追溯结果。"""

    class Algorithm(models.TextChoices):
        SINGLE_SEQUENCE = "single_sequence", "单台多任务顺序优化"
        MULTI_VNS = "multi_vns", "多台变邻域优化"

    batch_id = models.UUIDField("调度批次", default=uuid.uuid4, unique=True, editable=False)
    algorithm = models.CharField("优化算法", max_length=32, choices=Algorithm.choices)
    objective_name = models.CharField("目标函数", max_length=128)
    task_count = models.PositiveIntegerField("任务数", default=0)
    agv_count = models.PositiveIntegerField("使用 AGV 数", default=0)
    makespan_seconds = models.FloatField("完工时间(秒)", default=0)
    total_distance = models.FloatField("总里程(米)", default=0)
    total_wait_seconds = models.FloatField("总等待时间(秒)", default=0)
    iteration_count = models.PositiveIntegerField("搜索迭代次数", default=0)
    baseline_makespan_seconds = models.FloatField("基线完工时间(秒)", default=0)
    improvement_percent = models.FloatField("改进率(%)", default=0)
    assignments = models.JSONField("AGV 分配与顺序", default=dict, blank=True)
    metrics = models.JSONField("优化指标", default=dict, blank=True)
    created_at = models.DateTimeField("创建时间", auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "调度优化记录"
        verbose_name_plural = "调度优化记录"

    def __str__(self):
        return f"{self.get_algorithm_display()} - {self.task_count}任务/{self.agv_count}车"


class TransportTask(models.Model):
    class Priority(models.IntegerChoices):
        LOW = 1, "低"
        NORMAL = 2, "普通"
        HIGH = 3, "高"
        URGENT = 4, "紧急"

    class Status(models.TextChoices):
        PENDING = "pending", "待调度"
        ASSIGNED = "assigned", "已分配"
        IN_PROGRESS = "in_progress", "执行中"
        COMPLETED = "completed", "已完成"
        FAILED = "failed", "失败"
        CANCELLED = "cancelled", "已取消"

    task_no = models.CharField("任务编号", max_length=32, unique=True)
    pickup_node = models.ForeignKey(
        MapNode, on_delete=models.PROTECT, related_name="pickup_tasks", verbose_name="取货点"
    )
    dropoff_node = models.ForeignKey(
        MapNode, on_delete=models.PROTECT, related_name="dropoff_tasks", verbose_name="放货点"
    )
    cargo_name = models.CharField("货物名称", max_length=64)
    cargo_weight = models.FloatField("货物重量(kg)", validators=[MinValueValidator(0.1)])
    priority = models.PositiveSmallIntegerField(
        "优先级", choices=Priority.choices, default=Priority.NORMAL
    )
    status = models.CharField("状态", max_length=20, choices=Status.choices, default=Status.PENDING)
    assigned_agv = models.ForeignKey(
        AGV, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="tasks", verbose_name="执行 AGV",
    )
    schedule_run = models.ForeignKey(
        ScheduleRun, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="tasks", verbose_name="调度优化记录",
    )
    sequence_number = models.PositiveIntegerField("执行顺序", default=0)
    planned_wait_seconds = models.FloatField("计划避碰等待(秒)", default=0)
    optimization_details = models.JSONField("优化详情", default=dict, blank=True)
    route = models.JSONField("规划路线", default=dict, blank=True)
    total_distance = models.FloatField("总里程(米)", default=0)
    estimated_duration = models.FloatField("单任务耗时(秒)", default=0)
    scheduled_start = models.DateTimeField("计划开始", null=True, blank=True)
    scheduled_end = models.DateTimeField("计划完成", null=True, blank=True)
    started_at = models.DateTimeField("实际开始", null=True, blank=True)
    completed_at = models.DateTimeField("实际完成", null=True, blank=True)
    notes = models.CharField("备注", max_length=255, blank=True)
    created_at = models.DateTimeField("创建时间", auto_now_add=True)
    updated_at = models.DateTimeField("更新时间", auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["status", "priority", "created_at"]),
            models.Index(fields=["assigned_agv", "status"]),
            models.Index(fields=["schedule_run", "sequence_number"]),
        ]
        verbose_name = "运输任务"
        verbose_name_plural = "运输任务"

    def __str__(self):
        return f"{self.task_no} ({self.cargo_name})"

class DispatchRecord(models.Model):
    class Algorithm(models.TextChoices):
        SINGLE = "single", "单台调度"
        BATCH = "batch", "多台调度"

    task = models.ForeignKey(
        TransportTask, on_delete=models.CASCADE, related_name="dispatch_records", verbose_name="任务"
    )
    agv = models.ForeignKey(
        AGV, on_delete=models.SET_NULL, null=True,
        related_name="dispatch_records", verbose_name="AGV",
    )
    algorithm = models.CharField("调度算法", max_length=16, choices=Algorithm.choices)
    success = models.BooleanField("成功", default=True)
    message = models.CharField("调度说明", max_length=255, blank=True)
    route = models.JSONField("调度路线", default=dict, blank=True)
    distance = models.FloatField("调度里程(米)", default=0)
    estimated_duration = models.FloatField("预计耗时(秒)", default=0)
    created_at = models.DateTimeField("调度时间", auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "调度记录"
        verbose_name_plural = "调度记录"

    def __str__(self):
        return f"{self.task.task_no} -> {self.agv.code if self.agv else '未分配'}"
