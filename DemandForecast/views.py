from __future__ import annotations

import json

import pandas as pd
from django.db.models import Avg, Sum
from rest_framework import mixins, status, viewsets
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import DemandForecastRun
from .serializers import DemandForecastRunSerializer
from .services import (
    ALL_MODELS,
    MODEL_LABELS,
    DemandForecastError,
    forecast_demand,
    generate_order_series,
)


def _history_from_request(request) -> tuple[list[dict], str]:
    upload = request.FILES.get("file") or request.FILES.get("csv")
    if upload is not None:
        try:
            frame = pd.read_csv(upload)
        except Exception as exc:  # pragma: no cover
            raise DemandForecastError(f"CSV 读取失败：{exc}") from exc
        return frame.to_dict(orient="records"), DemandForecastRun.Source.UPLOAD

    raw_history = request.data.get("history")
    if raw_history is not None:
        if isinstance(raw_history, str):
            try:
                raw_history = json.loads(raw_history)
            except ValueError as exc:
                raise DemandForecastError("history 必须是 JSON 数组。") from exc
        if not isinstance(raw_history, list):
            raise DemandForecastError("history 必须是数组。")
        return raw_history, DemandForecastRun.Source.MANUAL

    return generate_order_series(), DemandForecastRun.Source.DATASET


class ForecastOverviewView(APIView):
    def get(self, request):
        total = DemandForecastRun.objects.count()
        average_mape = DemandForecastRun.objects.aggregate(value=Avg("mape"))["value"] or 0
        predicted_total = (
            DemandForecastRun.objects.aggregate(value=Sum("predicted_total"))["value"] or 0
        )
        return Response(
            {
                "total": total,
                "average_mape": round(average_mape, 4),
                "predicted_total": round(predicted_total, 2),
                "available_models": [
                    {"value": "auto", "label": "自动选择"},
                    *[
                        {"value": name, "label": MODEL_LABELS[name]}
                        for name in ALL_MODELS
                    ],
                ],
                "recent_runs": DemandForecastRunSerializer(
                    DemandForecastRun.objects.all()[:8],
                    many=True,
                ).data,
            }
        )


class ForecastSampleView(APIView):
    def get(self, request):
        days = int(request.query_params.get("days", 240))
        series = generate_order_series(days=days)
        return Response(
            {
                "days": len(series),
                "series": series,
                "hint": "可将该序列作为 history 字段提交到 /api/forecast/predict/。",
            }
        )


class ForecastPredictView(APIView):
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request):
        try:
            history, source = _history_from_request(request)
            horizon = int(request.data.get("horizon", 14))
            model = str(request.data.get("model") or "auto")
            result = forecast_demand(history, horizon=horizon, model=model)
        except (DemandForecastError, TypeError, ValueError) as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        metrics = result["metrics"]
        record = DemandForecastRun.objects.create(
            model_name=(
                model
                if model in {choice[0] for choice in DemandForecastRun.ModelName.choices}
                else DemandForecastRun.ModelName.AUTO
            ),
            selected_model=result["selected_model"],
            source=source,
            horizon=result["horizon"],
            history_count=result["history_count"],
            predicted_total=result["predicted_total"],
            mae=metrics["mae"],
            rmse=metrics["rmse"],
            mape=metrics["mape"],
            history=result["history"],
            forecast=result["forecast"],
            metrics=result["candidate_metrics"],
        )
        return Response({"success": True, "record_id": record.id, **result})


class ForecastRunViewSet(
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.DestroyModelMixin,
    viewsets.GenericViewSet,
):
    queryset = DemandForecastRun.objects.all()
    serializer_class = DemandForecastRunSerializer
    ordering_fields = ["created_at", "mape", "predicted_total"]

    def get_queryset(self):
        queryset = DemandForecastRun.objects.all()
        model = self.request.query_params.get("selected_model")
        if model:
            queryset = queryset.filter(selected_model=model)
        return queryset
