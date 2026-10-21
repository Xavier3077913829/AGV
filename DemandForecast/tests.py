from django.test import TestCase
from rest_framework.test import APIClient

from .models import DemandForecastRun
from .services import forecast_demand, generate_order_series


class ForecastServiceTests(TestCase):
    def test_sample_series(self):
        series = generate_order_series(days=120)
        self.assertEqual(len(series), 120)
        self.assertIn("date", series[0])
        self.assertIn("order_count", series[0])

    def test_moving_average_forecast(self):
        result = forecast_demand(generate_order_series(days=120), horizon=7, model="moving_average")
        self.assertEqual(result["selected_model"], "moving_average")
        self.assertEqual(len(result["forecast"]), 7)
        self.assertGreater(result["predicted_total"], 0)

    def test_auto_forecast_returns_candidate_metrics(self):
        result = forecast_demand(generate_order_series(days=120), horizon=7, model="auto")
        self.assertIn(result["selected_model"], {"moving_average", "holt_winters", "arima"})
        self.assertTrue(result["candidate_metrics"])


class ForecastApiTests(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.host = {"HTTP_HOST": "localhost"}

    def test_sample_endpoint(self):
        response = self.client.get("/api/forecast/sample/?days=90", **self.host)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["days"], 90)

    def test_predict_and_overview(self):
        response = self.client.post(
            "/api/forecast/predict/",
            {"horizon": 7, "model": "moving_average"},
            format="json",
            **self.host,
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["success"])
        self.assertEqual(response.data["horizon"], 7)
        self.assertEqual(len(response.data["forecast"]), 7)
        self.assertTrue(DemandForecastRun.objects.filter(id=response.data["record_id"]).exists())

        overview = self.client.get("/api/forecast/overview/", **self.host)
        self.assertEqual(overview.status_code, 200)
        self.assertEqual(overview.data["total"], 1)

    def test_invalid_horizon_returns_400(self):
        response = self.client.post(
            "/api/forecast/predict/",
            {"horizon": 500, "model": "moving_average"},
            format="json",
            **self.host,
        )
        self.assertEqual(response.status_code, 400)


class ForecastModelNameTests(TestCase):
    def test_model_name_includes_lightgbm(self):
        from .models import DemandForecastRun

        values = dict(DemandForecastRun.ModelName.choices).values()
        self.assertIn("LightGBM", "".join(values))
        self.assertIn("lightgbm", dict(DemandForecastRun.ModelName.choices))
