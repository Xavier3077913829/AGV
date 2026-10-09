from rest_framework import serializers

from .models import DemandForecastRun


class DemandForecastRunSerializer(serializers.ModelSerializer):
    model_display = serializers.CharField(source="get_model_name_display", read_only=True)
    source_display = serializers.CharField(source="get_source_display", read_only=True)

    class Meta:
        model = DemandForecastRun
        fields = [
            "id",
            "model_name",
            "model_display",
            "selected_model",
            "source",
            "source_display",
            "horizon",
            "history_count",
            "predicted_total",
            "mae",
            "rmse",
            "mape",
            "history",
            "forecast",
            "metrics",
            "created_at",
        ]
        read_only_fields = fields
