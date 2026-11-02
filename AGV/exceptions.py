# -*- coding: utf-8 -*-
"""DRF 统一异常处理：让接口在任何情况下都返回 JSON，而不是 Django 的 HTML 错误页。"""
from __future__ import annotations

import logging

from rest_framework.response import Response
from rest_framework.views import exception_handler

logger = logging.getLogger(__name__)


def api_exception_handler(exc, context):
    """先交给 DRF 处理已知异常；未捕获异常记录日志并返回结构化 500。"""
    response = exception_handler(exc, context)
    if response is not None:
        return response

    logger.exception("未处理的 API 异常", exc_info=exc)
    return Response(
        {"detail": "服务器内部错误，请查看后端日志。"},
        status=500,
    )
