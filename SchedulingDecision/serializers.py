from time import time

from rest_framework import serializers

from .models import AGV, DispatchRecord, MapEdge, MapNode, MapObstacle, ScheduleRun, TransportTask


class MapNodeSerializer(serializers.ModelSerializer):
    node_type_display = serializers.CharField(source="get_node_type_display", read_only=True)

    class Meta:
        model = MapNode
        fields = [
            "id", "code", "name", "node_type", "node_type_display", "x", "y",
            "is_active", "created_at",
        ]
        read_only_fields = ["created_at"]


class MapObstacleSerializer(serializers.ModelSerializer):
    class Meta:
        model = MapObstacle
        fields = ["id", "code", "name", "x", "y", "width", "height", "is_active", "created_at"]
        read_only_fields = ["created_at"]

class MapEdgeSerializer(serializers.ModelSerializer):
    start_node_detail = MapNodeSerializer(source="start_node", read_only=True)
    end_node_detail = MapNodeSerializer(source="end_node", read_only=True)

    class Meta:
        model = MapEdge
        fields = [
            "id", "start_node", "start_node_detail", "end_node", "end_node_detail",
            "distance", "speed_limit", "bidirectional", "is_active", "created_at",
        ]
        read_only_fields = ["created_at"]

    def validate(self, attrs):
        start = attrs.get("start_node", getattr(self.instance, "start_node", None))
        end = attrs.get("end_node", getattr(self.instance, "end_node", None))
        if start and end and start == end:
            raise serializers.ValidationError("道路起点和终点不能相同。")
        return attrs


class AGVSerializer(serializers.ModelSerializer):
    status_display = serializers.CharField(source="get_status_display", read_only=True)
    current_node_detail = MapNodeSerializer(source="current_node", read_only=True)
    active_task_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = AGV
        fields = [
            "id", "code", "name", "model_name", "status", "status_display",
            "current_node", "current_node_detail", "battery_percent",
            "payload_capacity", "speed", "active_task_count",
            "last_heartbeat", "created_at",
        ]
        read_only_fields = ["last_heartbeat", "created_at"]


class TransportTaskSerializer(serializers.ModelSerializer):
    status_display = serializers.CharField(source="get_status_display", read_only=True)
    priority_display = serializers.CharField(source="get_priority_display", read_only=True)
    pickup_node_detail = MapNodeSerializer(source="pickup_node", read_only=True)
    dropoff_node_detail = MapNodeSerializer(source="dropoff_node", read_only=True)
    assigned_agv_detail = AGVSerializer(source="assigned_agv", read_only=True)

    class Meta:
        model = TransportTask
        fields = [
            "id", "task_no", "pickup_node", "pickup_node_detail", "dropoff_node",
            "dropoff_node_detail", "cargo_name", "cargo_weight", "priority",
            "priority_display", "status", "status_display", "assigned_agv",
            "assigned_agv_detail", "schedule_run", "sequence_number", "planned_wait_seconds", "optimization_details", "route", "total_distance", "estimated_duration",
            "scheduled_start", "scheduled_end", "started_at", "completed_at",
            "notes", "created_at", "updated_at",
        ]
        read_only_fields = [
            "status", "assigned_agv", "schedule_run", "sequence_number", "planned_wait_seconds", "optimization_details", "route", "total_distance", "estimated_duration",
            "scheduled_start", "scheduled_end", "started_at", "completed_at",
            "created_at", "updated_at",
        ]
        extra_kwargs = {"task_no": {"required": False, "allow_blank": True}}

    def validate(self, attrs):
        pickup = attrs.get("pickup_node", getattr(self.instance, "pickup_node", None))
        dropoff = attrs.get("dropoff_node", getattr(self.instance, "dropoff_node", None))
        if pickup and dropoff and pickup == dropoff:
            raise serializers.ValidationError("取货点和放货点不能相同。")
        return attrs

    def create(self, validated_data):
        if not validated_data.get("task_no"):
            validated_data["task_no"] = f"TASK-{int(time() * 1000)}"
        return super().create(validated_data)


class DispatchRecordSerializer(serializers.ModelSerializer):
    algorithm_display = serializers.CharField(source="get_algorithm_display", read_only=True)
    task_no = serializers.CharField(source="task.task_no", read_only=True)
    agv_code = serializers.CharField(source="agv.code", read_only=True, allow_null=True)

    class Meta:
        model = DispatchRecord
        fields = [
            "id", "task", "task_no", "agv", "agv_code", "algorithm",
            "algorithm_display", "success", "message", "route", "distance",
            "estimated_duration", "created_at",
        ]


class ScheduleRunSerializer(serializers.ModelSerializer):
    algorithm_display = serializers.CharField(source="get_algorithm_display", read_only=True)

    class Meta:
        model = ScheduleRun
        fields = [
            "id", "batch_id", "algorithm", "algorithm_display", "objective_name",
            "task_count", "agv_count", "makespan_seconds", "total_distance",
            "total_wait_seconds", "iteration_count", "baseline_makespan_seconds",
            "improvement_percent", "assignments", "metrics", "created_at",
        ]