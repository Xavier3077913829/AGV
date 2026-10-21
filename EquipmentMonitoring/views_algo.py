# -*- coding: utf-8 -*-
"""设备监测 —— 算法增强接口（PyOD 异常检测 / tsfresh 自动特征）

路由（挂在 /api/equipment/algo/ 下）::

    GET  status/      依赖可用性
    POST detect/      PyOD 无监督异常检测
    POST features/    tsfresh 自动特征摘要
    POST compare/     RF 基线 vs PyOD 多模型对比
    POST diagnose/    原 RF 诊断 + PyOD 异常分数（融合结果）
"""
from __future__ import annotations

import json

import pandas as pd
from rest_framework import status
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .services import (
    EquipmentMonitoringError,
    compare_datasets,
    load_dataset_records,
)
from .services_algo import (
    binarize_labels,
    compare_detectors,
    detect_anomalies,
    diagnose_with_algo,
    features_summary,
    find_label_column,
    pick_sensor_columns,
    pyod_available,
    tsfresh_available,
)

PARSERS = (JSONParser, MultiPartParser, FormParser)


def _frame_from_request(request) -> pd.DataFrame:
    upload = request.FILES.get("file") or request.FILES.get("csv")
    if upload is not None:
        return pd.read_csv(upload)

    records = request.data.get("records")
    if isinstance(records, str):
        try:
            records = json.loads(records)
        except ValueError as exc:
            raise EquipmentMonitoringError("records 必须是 JSON 数组。") from exc
    if isinstance(records, list) and records:
        return pd.DataFrame(records)

    dataset = str(request.data.get("dataset") or "").strip()
    if dataset:
        limit = int(request.data.get("limit") or 500)
        return pd.DataFrame(load_dataset_records(dataset, limit=limit))

    raise EquipmentMonitoringError("请提供 file / records / dataset 之一")


class AlgoStatusView(APIView):
    """GET /api/equipment/algo/status/"""

    def get(self, request):
        available = {}
        for library in ("lightgbm", "xgboost", "imblearn"):
            try:
                __import__(library)
                available[library] = True
            except Exception:
                available[library] = False
        return Response({
            "pyod": pyod_available(),
            "tsfresh": tsfresh_available(),
            "classifier_libs": available,
            "install": "pip install ./vendor/pyod && pip install tsfresh",
        })


class AnomalyDetectView(APIView):
    """POST /api/equipment/algo/detect/   {file|records|dataset, method}"""

    parser_classes = PARSERS

    def post(self, request):
        if not pyod_available():
            return Response({"detail": "pyod 未安装，请先 pip install ./vendor/pyod"},
                            status=status.HTTP_503_SERVICE_UNAVAILABLE)
        try:
            frame = _frame_from_request(request)
            method = str(request.data.get("method") or "ECOD").upper()
            scores, labels, model = detect_anomalies(frame, method=method)
        except EquipmentMonitoringError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as exc:                     # pragma: no cover
            return Response({"detail": f"{type(exc).__name__}: {exc}"},
                            status=status.HTTP_400_BAD_REQUEST)
        return Response({
            "method": method,
            "rows": int(len(scores)),
            "flagged": int(sum(labels)),
            "scores": [round(float(s), 4) for s in scores],
            "labels": [int(v) for v in labels],
        })


class FeatureExtractView(APIView):
    """POST /api/equipment/algo/features/   {file|records|dataset, window}"""

    parser_classes = PARSERS

    def post(self, request):
        if not tsfresh_available():
            return Response({"detail": "tsfresh 未安装，请先 pip install tsfresh"},
                            status=status.HTTP_503_SERVICE_UNAVAILABLE)
        try:
            frame = _frame_from_request(request)
            window = int(request.data.get("window") or 64)
            summary = features_summary(frame, window=window)
        except EquipmentMonitoringError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as exc:                     # pragma: no cover
            return Response({"detail": f"{type(exc).__name__}: {exc}"},
                            status=status.HTTP_400_BAD_REQUEST)
        return Response(summary)


class DetectorCompareView(APIView):
    """POST /api/equipment/algo/compare/   {file|records|dataset, label}"""

    parser_classes = PARSERS

    def post(self, request):
        try:
            frame = _frame_from_request(request)
            label = request.data.get("label") or None
            table = compare_detectors(frame, label_col=label)
        except EquipmentMonitoringError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as exc:                     # pragma: no cover
            return Response({"detail": f"{type(exc).__name__}: {exc}"},
                            status=status.HTTP_400_BAD_REQUEST)
        lab = label or find_label_column(frame)
        return Response({
            "label_column": lab,
            "features": pick_sensor_columns(frame, lab),
            "rows": table.to_dict(orient="records"),
        })


class DiagnosePlusView(APIView):
    """POST /api/equipment/algo/diagnose/   {file|records|dataset, equipment_code, method}"""

    parser_classes = PARSERS

    def post(self, request):
        try:
            frame = _frame_from_request(request)
            records = frame.to_dict(orient="records")
            data_source = str(request.data.get("dataset") or "ai4i_equipment")
            result = diagnose_with_algo(
                records,
                equipment_code=str(request.data.get("equipment_code") or "CNC-01"),
                data_source=data_source,
                method=str(request.data.get("method") or "ECOD").upper(),
            )
        except EquipmentMonitoringError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as exc:                     # pragma: no cover
            return Response({"detail": f"{type(exc).__name__}: {exc}"},
                            status=status.HTTP_400_BAD_REQUEST)
        return Response(result)


class DatasetCompareView(APIView):
    """GET /api/equipment/algo/dataset-compare/

    对比不同训练数据集的 worn 类效果（ai4i 46 条 vs NASA 铣削 760 条）。
    """

    parser_classes = (JSONParser,)

    def get(self, request):
        try:
            rows = compare_datasets()
        except Exception as exc:  # noqa: BLE001
            return Response({"detail": f"{type(exc).__name__}: {exc}"},
                            status=status.HTTP_400_BAD_REQUEST)
        best = max(
            (r for r in rows if "error" not in r),
            key=lambda r: (r.get("worn_f1") or 0),
            default=None,
        )
        return Response({
            "success": True,
            "datasets": rows,
            "best_worn": best.get("dataset") if best else None,
        })
