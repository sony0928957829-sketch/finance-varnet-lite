from pathlib import Path
import unittest
from datetime import date
import json
import pandas as pd

from src.features.labels import range_label_columns
from src.main import provider_price_settings, run_pipeline, years_before
from src.utils.config import DATA_DIR, load_config


class PipelineSmokeTest(unittest.TestCase):
    def test_five_year_history_start(self):
        self.assertEqual(years_before(date(2026, 6, 11), 5), date(2021, 6, 11))

    def test_yfinance_settings_come_from_data_sources_config(self):
        config = load_config("data_sources.yaml")

        symbols, history_years = provider_price_settings(config, "yfinance")

        self.assertEqual(symbols, ["NVDA", "TSLA", "AMD", "BTC-USD"])
        self.assertEqual(history_years, 5)

    def test_mock_pipeline_runs(self):
        path = run_pipeline(mode="mock", start="2025-01-01", end="2025-12-31")

        self.assertTrue(Path(path).exists())
        report = Path(path).read_text(encoding="utf-8")
        self.assertTrue(report.startswith("# VARnet-lite"))
        self.assertIn("不構成買賣建議", report)

        health_path = Path(path).with_name(
            Path(path).name.replace("_market_report.md", "_data_health.json")
        )
        health = json.loads(health_path.read_text(encoding="utf-8"))
        self.assertNotEqual(health["status"], "error")

        features = pd.read_parquet(DATA_DIR / "features" / "features_mock.parquet")
        labels = pd.read_parquet(DATA_DIR / "labels" / "labels_mock.parquet")
        label_columns = range_label_columns()

        self.assertTrue(set(label_columns).isdisjoint(features.columns))
        self.assertTrue(set(label_columns).issubset(labels.columns))


if __name__ == "__main__":
    unittest.main()
