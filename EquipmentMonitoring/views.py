from __future__ import annotations

import json

import pandas as pd
from django.db.models import Avg, Count
from rest_framework import mixins, status, viewsets
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import EquipmentDiagnosisRecord
from .serializers import (
    EquipmentDiagnosisRecordListSerializer,
    EquipmentDiagnosisRecordSerializer,
)
from .services import (
    EquipmentMonitoringError,
    available_datasets,
    diagnose_equipment,
    generate_demo_series,
    get_model_bundle,
    load_dataset_records,
)


def _records_from_request(request) -> tuple[list[dict], str]:
    upload = request.FILES.get("file") or request.FILES.get("csv")
    if upload is not None:
        try:
            frame = pd.read_csv(upload)
        except Exception as exc:  # pragma: no cover - pandas 具体异常类型随版本变化
            raise EquipmentMonitoringError(f"CSV 读取失败：{exc}") from exc
        return frame.to_dict(orient="records"), EquipmentDiagnosisRecord.Source.UPLOAD

    raw_records = request.data.get("records")
    if raw_records is not None:
        if isinstance(raw_records, str):
            try:
                raw_records = json.loads(raw_records)
            except ValueError as exc:
                raise EquipmentMonitoringError("records 必须是 JSON 数组。") from exc
        if not isinstance(raw_records, list):
            raise EquipmentMonitoringError("records 必须是数组。")
        return raw_records, EquipmentDiagnosisRecord.Source.MANUAL

    dataset = str(request.data.get("dataset") or "").strip()
    if dataset:
        # 样本数在 services 里收敛到 DATASET_LIMIT_MAX，避免一次取走上万行
        return (
            load_dataset_records(dataset, limit=request.data.get("limit")),
            EquipmentDiagnosisRecord.Source.DATASET,
        )

    scenario = str(request.data.get("sample") or request.data.get("scenario") or "")
    if scenario:
        return (
            generate_demo_series(scenario=scenario),
            EquipmentDiagnosisRecord.Source.DATASET,
        )

    raise EquipmentMonitoringError(
        "请提供 records 数组、CSV 文件，或 sample=normal/worn/fault、dataset=ai4i_equipment。"
    )


class EquipmentOverviewView(APIView):
    def get(self, request):
        total = EquipmentDiagnosisRecord.objects.count()
        source = str(request.query_params.get("data_source") or "synthetic")
        state_counts = {
            item["predicted_state"]: item["count"]
            for item in EquipmentDiagnosisRecord.objects.values("predicted_state").annotate(
                count=Count("id")
            )
        }
        average_confidence = (
            EquipmentDiagnosisRecord.objects.aggregate(value=Avg("confidence"))["value"] or 0
        )
        try:
            model_metrics = get_model_bundle(source)["metrics"]
        except EquipmentMonitoringError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(
            {
                "total": total,
                "state_counts": state_counts,
                "average_confidence": round(average_confidence, 4),
                "model_metrics": model_metrics,
                "available_datasets": available_datasets(),
                "recent_records": EquipmentDiagnosisRecordListSerializer(
                    EquipmentDiagnosisRecord.objects.all()[:8],
                    many=True,
                ).data,
            }
        )


class EquipmentDatasetView(APIView):
    """列出 datasets/equipment 下的真实数据集（来源、规模、标签分布）。"""

    def get(self, request):
        return Response(
            {
                "datasets": available_datasets(),
                "hint": "在 /api/equipment/sample/?dataset=<name>&limit=200 取样本，或给 /api/equipment/diagnose/ 传 dataset 与 data_source。",
            }
        )


class EquipmentSampleView(APIView):
    def get(self, request):
        dataset = str(request.query_params.get("dataset") or "").strip()
        if dataset:
            try:
                series = load_dataset_records(
                    dataset, limit=request.query_params.get("limit")
                )
            except EquipmentMonitoringError as exc:
                return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
            source = next(
                (item["source"] for item in available_datasets() if item["name"] == dataset),
                "本地数据集",
            )
            return Response(
                {
                    "dataset": dataset,
                    "source": source,
                    "sample_count": len(series),
                    "series": series,
                    "hint": "真实数据集样本；可直接提交到 /api/equipment/diagnose/ 的 records 字段（建议同时传 data_source）。",
                }
            )

        scenario = str(request.query_params.get("scenario", "worn"))
        series = generate_demo_series(scenario=scenario)
        return Response(
            {
                "scenario": scenario,
                "sample_count": len(series),
                "series": series,
                "hint": "可直接将该序列提交到 /api/equipment/diagnose/ 的 records 字段。",
            }
        )


class EquipmentModelView(APIView):
    def get(self, request):
        source = str(request.query_params.get("data_source") or "synthetic")
        try:
            bundle = get_model_bundle(source)
        except EquipmentMonitoringError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        return Response(
            {
                "algorithm": "StandardScaler + RandomForestClassifier",
                "data_source": source,
                "features": bundle["features"],
                "feature_importance": bundle["feature_importance"],
                "metrics": bundle["metrics"],
                "available_datasets": available_datasets(),
            }
        )


class EquipmentDiagnoseView(APIView):
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request):
        try:
            records, source = _records_from_request(request)
            equipment_code = str(request.data.get("equipment_code") or "CNC-01")
            data_source = str(request.data.get("data_source") or "synthetic")
            result = diagnose_equipment(
                records, equipment_code=equipment_code, data_source=data_source
            )
        except EquipmentMonitoringError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        record = EquipmentDiagnosisRecord.objects.create(
            equipment_code=equipment_code,
            source=source,
            sample_count=result["sample_count"],
            predicted_state=result["predicted_state"],
            confidence=result["confidence"],
            state_counts=result["state_distribution"],
            rule_alerts=result["rule_alerts"],
            features_summary=result["feature_summary"],
            series=result["series"],
            model_metrics=result["model_metrics"],
        )
        return Response(
            {
                "success": True,
                "record_id": record.id,
                **result,
            }
        )


class EquipmentRecordViewSet(
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.DestroyModelMixin,
    viewsets.GenericViewSet,
):
    queryset = EquipmentDiagnosisRecord.objects.all()
    serializer_class = EquipmentDiagnosisRecordSerializer
    ordering_fields = ["created_at", "confidence", "sample_count"]

    def get_serializer_class(self):
        # 列表接口不返回完整采样序列：一条上万行的记录会把分页响应撑到上百 MB
        if self.action == "list":
            return EquipmentDiagnosisRecordListSerializer
        return EquipmentDiagnosisRecordSerializer

    def get_queryset(self):
        queryset = EquipmentDiagnosisRecord.objects.all()
        for field in ("equipment_code", "predicted_state", "source"):
            value = self.request.query_params.get(field)
            if value:
                queryset = queryset.filter(**{field: value})
        return queryset
