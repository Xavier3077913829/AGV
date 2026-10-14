import pandas as pd
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

class ForecastDatasetApiTests(TestCase):
    """真实订单序列接口与频率推断。"""

    def setUp(self):
        self.client = APIClient()
        self.host = {"HTTP_HOST": "localhost"}

    def test_series_list_and_sample(self):
        listing = self.client.get("/api/forecast/series/", **self.host)
        self.assertEqual(listing.status_code, 200)
        names = [item["name"] for item in listing.data["series"]]
        self.assertIn("online_retail_daily", names)

        sample = self.client.get(
            "/api/forecast/sample/?series=online_retail_daily", **self.host
        )
        self.assertEqual(sample.status_code, 200)
        self.assertEqual(sample.data["series_name"], "online_retail_daily")
        self.assertEqual(sample.data["days"], len(sample.data["series"]))

    def test_series_name_cannot_escape_series_dir(self):
        response = self.client.get(
            "/api/forecast/sample/?series=../equipment/ai4i_equipment", **self.host
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("不存在", response.data["detail"])

    def test_unknown_series_returns_400(self):
        response = self.client.post(
            "/api/forecast/predict/",
            {"series": "not-a-series"},
            format="json",
            **self.host,
        )
        self.assertEqual(response.status_code, 400)

    def test_predict_with_real_series(self):
        response = self.client.post(
            "/api/forecast/predict/",
            {"series": "online_retail_daily", "horizon": 7, "model": "moving_average"},
            format="json",
            **self.host,
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["frequency"], "D")
        self.assertEqual(response.data["history_count"], 374)
        self.assertEqual(len(response.data["forecast"]), 7)

    def test_weekly_series_keeps_weekday(self):
        start = pd.Timestamp("2011-01-03")  # 周一
        history = [
            {
                "date": (start + pd.Timedelta(days=7 * index)).strftime("%Y-%m-%d"),
                "order_count": 50 + index % 7,
            }
            for index in range(60)
        ]
        result = forecast_demand(history, horizon=5, model="moving_average")
        self.assertEqual(result["frequency"], "W")
        dates = pd.to_datetime([item["date"] for item in result["forecast"]])
        self.assertTrue(all(date.dayofweek == 0 for date in dates))
        self.assertTrue(
            all(delta == pd.Timedelta(days=7) for delta in pd.Series(dates).diff().dropna())
        )
