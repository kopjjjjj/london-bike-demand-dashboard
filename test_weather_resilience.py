"""Regression tests for weather-service degradation paths."""

from pathlib import Path
from unittest import TestCase, mock
import tempfile

import pandas as pd

import app
import open_meteo


class WeatherFallbackTests(TestCase):
    def test_both_panels_remain_populated_during_outage(self):
        failure = RuntimeError("simulated Open-Meteo outage")
        with (
            mock.patch.object(app, "open_meteo", side_effect=failure),
            mock.patch.object(app, "open_meteo_history", side_effect=failure),
        ):
            result = app.update_weather_predictions(0, "London")

        status, history_figure, history_rows, _, forecast_figure, forecast_rows, _ = result
        self.assertEqual(len(history_rows), 7)
        self.assertEqual(len(forecast_rows), 5)
        self.assertTrue(all(row["predicted_hires"] >= 0 for row in history_rows))
        self.assertTrue(all(row["predicted_hires"] >= 0 for row in forecast_rows))
        self.assertTrue(history_figure.data)
        self.assertTrue(forecast_figure.data)
        self.assertIn("not observed/live weather", status)

    def test_missing_model_field_is_filled_from_climatology(self):
        partial = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=2),
                "temp": [5.0, 6.0],
                "precip": [0.0, 1.0],
                "windspeed": [20.0, 21.0],
                "visibility": [None, 18.0],
            }
        )
        partial.attrs["source"] = "test service"

        weather, error = app.usable_weather(
            lambda: partial, pd.date_range("2026-01-01", periods=2)
        )

        self.assertIsNone(error)
        self.assertFalse(weather[app.MODEL_VARIABLES].isna().any().any())
        self.assertIn("uvindex", weather.attrs["note"])
        self.assertIn("visibility", weather.attrs["note"])


class WeatherCacheTests(TestCase):
    def test_successful_response_round_trips_through_cache(self):
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2026-01-01", periods=2),
                "temp": [5.0, 6.0],
            }
        )
        frame.attrs["location"] = "London, United Kingdom"

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(open_meteo, "CACHE_DIR", Path(directory)):
                open_meteo._write_cache("forecast", "london:2", frame)
                cached = open_meteo._read_cache("forecast", "london:2", 3600)

        self.assertIsNotNone(cached)
        self.assertEqual(len(cached), 2)
        self.assertEqual(cached.attrs["source"], "cache")
        self.assertEqual(cached.attrs["location"], "London, United Kingdom")

