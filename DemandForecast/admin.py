from django.contrib import admin

from .models import DemandForecastRun


@admin.register(DemandForecastRun)
class DemandForecastRunAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "model_name",
        "selected_model",
        "source",
        "horizon",
        "predicted_total",
        "mape",
        "created_at",
    )
    list_filter = ("model_name", "selected_model", "source")
