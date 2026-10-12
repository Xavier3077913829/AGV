from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import (
    ForecastOverviewView,
    ForecastPredictView,
    ForecastRunViewSet,
    ForecastSampleView,
)


router = DefaultRouter()
router.register("runs", ForecastRunViewSet, basename="forecast-run")

urlpatterns = [
    path("", include(router.urls)),
    path("overview/", ForecastOverviewView.as_view(), name="forecast-overview"),
    path("sample/", ForecastSampleView.as_view(), name="forecast-sample"),
    path("predict/", ForecastPredictView.as_view(), name="forecast-predict"),
]
