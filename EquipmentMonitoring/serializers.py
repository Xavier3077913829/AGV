from rest_framework import serializers

from .models import EquipmentDiagnosisRecord


class EquipmentDiagnosisRecordSerializer(serializers.ModelSerializer):
    source_display = serializers.CharField(source="get_source_display", read_only=True)
    state_display = serializers.CharField(source="get_predicted_state_display", read_only=True)

    class Meta:
        model = EquipmentDiagnosisRecord
        fields = [
            "id",
            "equipment_code",
            "source",
            "source_display",
            "sample_count",
            "predicted_state",
            "state_display",
            "confidence",
            "state_counts",
            "rule_alerts",
            "features_summary",
            "series",
            "model_metrics",
            "created_at",
        ]
        read_only_fields = fields
