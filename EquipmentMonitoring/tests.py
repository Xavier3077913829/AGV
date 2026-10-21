from django.test import TestCase
from rest_framework.test import APIClient

from .models import EquipmentDiagnosisRecord
from .services import diagnose_equipment, generate_demo_series, get_model_bundle


class EquipmentServiceTests(TestCase):
    def test_model_validation_is_reasonable(self):
        metrics = get_model_bundle()["metrics"]
        self.assertGreaterEqual(metrics["accuracy"], 0.9)
        self.assertGreaterEqual(metrics["f1_macro"], 0.9)

    def test_model_bundle_reports_algorithm_and_per_class_metrics(self):
        bundle = get_model_bundle(model_type="random_forest", use_smote=False)
        self.assertEqual(bundle["model_type"], "random_forest")
        self.assertIn("f1_per_class", bundle["metrics"])
        self.assertIn("algorithm", bundle["metrics"])

    def test_diagnose_accepts_model_type(self):
        result = diagnose_equipment(
            generate_demo_series("worn", points=50),
            model_type="random_forest",
            use_smote=False,
        )
        self.assertEqual(result["model_type"], "random_forest")
        self.assertIn(result["predicted_state"], {"worn", "fault"})

    def test_worn_sample_is_detected(self):
        result = diagnose_equipment(generate_demo_series("worn", points=50))
        self.assertEqual(result["sample_count"], 50)
        self.assertIn(result["predicted_state"], {"worn", "fault"})
        self.assertIn("feature_importance", result)

    def test_fault_sample_contains_rule_alerts(self):
        result = diagnose_equipment(generate_demo_series("fault", points=50))
        self.assertGreater(len(result["rule_alerts"]), 0)
        self.assertEqual(result["predicted_state"], "fault")


class EquipmentApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.host = {"HTTP_HOST": "localhost"}

    def test_sample_endpoint(self):
        response = self.client.get("/api/equipment/sample/?scenario=normal", **self.host)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["scenario"], "normal")
        self.assertGreater(response.data["sample_count"], 20)

    def test_diagnose_and_overview(self):
        response = self.client.post(
            "/api/equipment/diagnose/",
            {"sample": "worn", "equipment_code": "CNC-TEST"},
            format="json",
            **self.host,
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["success"])
        self.assertEqual(response.data["equipment_code"], "CNC-TEST")
        self.assertTrue(EquipmentDiagnosisRecord.objects.filter(id=response.data["record_id"]).exists())

        overview = self.client.get("/api/equipment/overview/", **self.host)
        self.assertEqual(overview.status_code, 200)
        self.assertEqual(overview.data["total"], 1)

    def test_missing_records_returns_400(self):
        response = self.client.post(
            "/api/equipment/diagnose/",
            {},
            format="json",
            **self.host,
        )
        self.assertEqual(response.status_code, 400)
