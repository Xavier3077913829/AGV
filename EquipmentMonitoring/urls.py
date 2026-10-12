from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import (
    EquipmentDatasetView,
    EquipmentDiagnoseView,
    EquipmentModelView,
    EquipmentOverviewView,
    EquipmentRecordViewSet,
    EquipmentSampleView,
)


router = DefaultRouter()
router.register("records", EquipmentRecordViewSet, basename="equipment-record")

urlpatterns = [
    path("", include(router.urls)),
    path("overview/", EquipmentOverviewView.as_view(), name="equipment-overview"),
    path("datasets/", EquipmentDatasetView.as_view(), name="equipment-datasets"),
    path("sample/", EquipmentSampleView.as_view(), name="equipment-sample"),
    path("model/", EquipmentModelView.as_view(), name="equipment-model"),
    path("diagnose/", EquipmentDiagnoseView.as_view(), name="equipment-diagnose"),
]
