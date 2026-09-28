from django.contrib import admin

from .models import AGV, DispatchRecord, MapEdge, MapNode, ScheduleRun, TransportTask


@admin.register(MapNode)
class MapNodeAdmin(admin.ModelAdmin):
    list_display = ("code", "name", "node_type", "x", "y", "is_active")
    list_filter = ("node_type", "is_active")
    search_fields = ("code", "name")


@admin.register(MapEdge)
class MapEdgeAdmin(admin.ModelAdmin):
    list_display = ("start_node", "end_node", "distance", "speed_limit", "bidirectional", "is_active")
    list_filter = ("is_active", "bidirectional")
    autocomplete_fields = ("start_node", "end_node")


@admin.register(AGV)
class AGVAdmin(admin.ModelAdmin):
    list_display = ("code", "name", "status", "current_node", "battery_percent", "payload_capacity")
    list_filter = ("status", "model_name")
    search_fields = ("code", "name")


@admin.register(TransportTask)
class TransportTaskAdmin(admin.ModelAdmin):
    list_display = ("task_no", "cargo_name", "pickup_node", "dropoff_node", "priority", "status", "assigned_agv")
    list_filter = ("status", "priority")
    search_fields = ("task_no", "cargo_name")


@admin.register(DispatchRecord)
class DispatchRecordAdmin(admin.ModelAdmin):
    list_display = ("task", "agv", "algorithm", "success", "distance", "estimated_duration", "created_at")
    list_filter = ("algorithm", "success")
    readonly_fields = ("created_at",)


@admin.register(ScheduleRun)
class ScheduleRunAdmin(admin.ModelAdmin):
    list_display = (
        "batch_id", "algorithm", "task_count", "agv_count",
        "makespan_seconds", "improvement_percent", "created_at",
    )
    list_filter = ("algorithm",)
    readonly_fields = ("batch_id", "created_at", "metrics", "assignments")