from __future__ import annotations

import unittest

from src.utils.config import load_config


class DataSourcesConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config("data_sources.yaml")

    def test_all_dataset_routes_have_primary_and_fallback(self):
        providers = self.config["providers"]
        datasets = self.config["datasets"]

        for category, routes in datasets.items():
            for route_name, route in routes.items():
                with self.subTest(category=category, route=route_name):
                    self.assertIn("primary", route)
                    self.assertIn("fallback", route)
                    self.assertIsInstance(route["fallback"], list)
                    self.assertIn(route["primary"], providers)
                    for fallback in route["fallback"]:
                        self.assertIn(fallback, providers)

    def test_current_market_priorities(self):
        prices = self.config["datasets"]["prices"]

        self.assertEqual(prices["us_stock"]["primary"], "yfinance")
        self.assertEqual(prices["crypto"]["primary"], "yfinance")
        self.assertEqual(prices["taiwan_stock"]["primary"], "yfinance")
        self.assertIn("finmind", prices["taiwan_stock"]["fallback"])
        self.assertIn("twse", prices["taiwan_stock"]["fallback"])
        self.assertEqual(prices["taiwan_futures"]["primary"], "taifex")
        self.assertIn("finmind", prices["taiwan_futures"]["fallback"])
        options = self.config["datasets"]["derivatives"]["taiwan_options"]
        self.assertEqual(options["primary"], "finmind")
        self.assertIn("taifex", options["fallback"])

    def test_future_dataset_categories_are_reserved(self):
        datasets = self.config["datasets"]

        self.assertIn("news", datasets)
        self.assertIn("chip", datasets)
        self.assertIn("derivatives", datasets)
        self.assertIn("macro", datasets)
        self.assertIn("taiwan_options", datasets["derivatives"])
        self.assertIn("vix", datasets["macro"])
        self.assertIn("us_treasury_yield", datasets["macro"])
        self.assertIn("us_dollar_index", datasets["macro"])
        self.assertIn("foreign_exchange", datasets["macro"])


if __name__ == "__main__":
    unittest.main()
