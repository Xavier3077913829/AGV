# -*- coding: utf-8 -*-
"""制造视觉 —— 算法增强接口（Anomalib PatchCore / MVTec 数据集）

路由（挂在 /api/vision/algo/ 下）::

    GET  status/    依赖可用性
    POST dataset/   把项目数据转成 MVTec-AD 布局
    POST patchcore/ PatchCore 推理（需先训练）
"""
from __future__ import annotations

import os

from rest_framework import status
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.response import Response
from rest_framework.views import APIView

from .services_supervised import segment_crack, status as deepcrack_status
from .services_defect import classify_defect, status as defect_status
from .services_algo import (
    anomalib_available,
    build_dataset_from_project,
    collect_dataset_paths,
    predict_patchcore,
)

PARSERS = (JSONParser, MultiPartParser, FormParser)


class AlgoStatusView(APIView):
    """GET /api/vision/algo/status/"""

    def get(self, request):
        paths = collect_dataset_paths()
        return Response({
            "anomalib": anomalib_available(),
            "deepcrack": deepcrack_status(),
            "dataset": {k: len(v) for k, v in paths.items()},
            "install": "pip install -e ./vendor/anomalib",
        })


class DatasetBuildView(APIView):
    """POST /api/vision/algo/dataset/  → 生成 MVTec-AD 布局"""

    parser_classes = PARSERS

    def post(self, request):
        try:
            info = build_dataset_from_project(
                dst_root=request.data.get("dst_root") or None,
                category=str(request.data.get("category") or "mt_crack"),
            )
        except Exception as exc:                     # pragma: no cover
            return Response({"detail": f"{type(exc).__name__}: {exc}"},
                            status=status.HTTP_400_BAD_REQUEST)
        return Response(info)


class PatchcoreInferView(APIView):
    """POST /api/vision/algo/patchcore/   {file} 或 {images:[...]}"""

    parser_classes = PARSERS

    def post(self, request):
        if not anomalib_available():
            return Response({"detail": "anomalib 未安装，请先 pip install -e ./vendor/anomalib"},
                            status=status.HTTP_503_SERVICE_UNAVAILABLE)
        upload = request.FILES.get("file")
        if upload is None:
            return Response({"detail": "请上传图片（字段名 file）"},
                            status=status.HTTP_400_BAD_REQUEST)
        tmp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "datasets", "auto", "_incoming")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, os.path.basename(upload.name or "upload.jpg"))
        with open(tmp_path, "wb") as f:
            for chunk in upload.chunks():
                f.write(chunk)
        try:
            result = predict_patchcore([tmp_path],
                                       model_dir=request.data.get("model_dir") or None)
        except Exception as exc:
            return Response({"detail": f"{type(exc).__name__}: {exc}"},
                            status=status.HTTP_400_BAD_REQUEST)
        return Response({"results": result})


class DeepCrackSegmentView(APIView):
    """POST /api/vision/algo/deepcrack/   {file}  → 监督式裂纹分割"""

    parser_classes = PARSERS

    def post(self, request):
        upload = request.FILES.get("file")
        if upload is None:
            return Response({"detail": "请上传图片（字段名 file）"},
                            status=status.HTTP_400_BAD_REQUEST)
        tmp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "datasets", "auto", "_incoming")
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_path = os.path.join(tmp_dir, os.path.basename(upload.name or "upload.jpg"))
        with open(tmp_path, "wb") as f:
            for chunk in upload.chunks():
                f.write(chunk)
        try:
            result = segment_crack(tmp_path,
                                   device=request.data.get("device") or "cpu",
                                   threshold=float(request.data.get("threshold") or 0.5))
            result.pop("mask_png_bytes", None)      # 二进制不进 JSON
        except RuntimeError as exc:
            return Response({"detail": str(exc), **deepcrack_status()},
                            status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except Exception as exc:                    # noqa: BLE001
            return Response({"detail": f"{type(exc).__name__}: {exc}"},
                            status=status.HTTP_400_BAD_REQUEST)
        return Response(result)


class DefectClassifyView(APIView):
    """GET/POST /api/vision/algo/defect/   NEU 6 类表面缺陷识别。

    POST 参数：image（multipart 文件）或 sample（内置样例编码），
    返回 predicted_class / predicted_label / confidence / top3。
    """

    parser_classes = PARSERS

    def get(self, request):
        return Response(defect_status())

    def post(self, request):
        import time

        from .views import _resolve_input

        try:
            image, _source, _name = _resolve_input(request)
        except ValueError as exc:
            return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)

        started = time.perf_counter()
        try:
            result = classify_defect(image, device=request.data.get("device") or "cpu")
        except RuntimeError as exc:
            return Response({"detail": str(exc), **defect_status()},
                            status=status.HTTP_503_SERVICE_UNAVAILABLE)
        except Exception as exc:  # noqa: BLE001
            return Response({"detail": f"{type(exc).__name__}: {exc}"},
                            status=status.HTTP_400_BAD_REQUEST)

        result.update({
            "task_type": "defect",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        })
        return Response(result)
