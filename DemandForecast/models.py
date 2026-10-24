from django.db import models


class DemandForecastRun(models.Model):
    """一次订单需求预测的可追溯结果。"""

    class ModelName(models.TextChoices):
        AUTO = "auto", "自动选择"
        MOVING_AVERAGE = "moving_average", "移动平均"
        HOLT_WINTERS = "holt_winters", "Holt-Winters"
        ARIMA = "arima", "ARIMA"
        LIGHTGBM = "lightgbm", "LightGBM（日历+滞后）"

    class Source(models.TextChoices):
        DATASET = "dataset", "内置样例"
        UPLOAD = "upload", "CSV 上传"
        MANUAL = "manual", "手工序列"

    model_name = models.CharField(
        "预测算法", max_length=24, choices=ModelName.choices, default=ModelName.AUTO
    )
    selected_model = models.CharField("实际采用模型", max_length=24, blank=True)
    source = models.CharField(
        "数据来源", max_length=16, choices=Source.choices, default=Source.DATASET
    )
    horizon = models.PositiveIntegerField("预测期数", default=14)
    history_count = models.PositiveIntegerField("历史数据量", default=0)
    predicted_total = models.FloatField("预测总量", default=0)
    mae = models.FloatField("MAE", default=0)
    rmse = models.FloatField("RMSE", default=0)
    mape = models.FloatField("MAPE(%)", default=0)
    history = models.JSONField("历史序列", default=list, blank=True)
    forecast = models.JSONField("预测序列", default=list, blank=True)
    metrics = models.JSONField("模型对比指标", default=dict, blank=True)
    created_at = models.DateTimeField("创建时间", auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["selected_model", "created_at"])]
        verbose_name = "订单预测记录"
        verbose_name_plural = "订单预测记录"

    def __str__(self):
        return f"{self.get_model_name_display()} - 未来{self.horizon}期"
