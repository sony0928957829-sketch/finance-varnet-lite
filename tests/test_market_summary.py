from __future__ import annotations

import unittest

import pandas as pd

from src.report.market_summary import build_market_summary, classify_market_state


class MarketSummaryTest(unittest.TestCase):
    def test_high_risk_overrides_directional_state(self):
        self.assertEqual(classify_market_state(65.0, 80.0), "高波動")
        self.assertEqual(classify_market_state(50.0, 65.0), "偏多")
        self.assertEqual(classify_market_state(50.0, 35.0), "轉弱")

    def test_summary_ranks_symbols_without_strong_weak_overlap(self):
        latest = pd.DataFrame(
            [
                {
                    "symbol": "NVDA",
                    "market": "US",
                    "risk_score": 70.0,
                    "condition_score": 30.0,
                    "return_1d": -0.03,
                },
                {
                    "symbol": "AMD",
                    "market": "US",
                    "risk_score": 60.0,
                    "condition_score": 80.0,
                    "return_1d": -0.01,
                },
                {
                    "symbol": "TSLA",
                    "market": "US",
                    "risk_score": 75.0,
                    "condition_score": 20.0,
                    "return_1d": -0.04,
                },
                {
                    "symbol": "BTC-USD",
                    "market": "CRYPTO",
                    "risk_score": 55.0,
                    "condition_score": 65.0,
                    "return_1d": 0.02,
                },
            ]
        )
        health = {
            "status": "warning",
            "summary": {"warning_count": 1},
        }

        summary = build_market_summary(latest, health)

        self.assertEqual(summary["market_state"], "高波動")
        self.assertEqual(summary["abnormal_symbols"][0], "TSLA")
        self.assertEqual(summary["strong_symbols"], ["AMD", "BTC-USD"])
        self.assertEqual(summary["weak_symbols"], ["TSLA", "NVDA"])
        self.assertTrue(
            set(summary["strong_symbols"]).isdisjoint(summary["weak_symbols"])
        )
        self.assertEqual(summary["data_health"], "警告（1 項）")
        self.assertTrue(any("跨市場背離" in point for point in summary["watch_points"]))

    def test_empty_data_is_explicitly_insufficient(self):
        summary = build_market_summary(pd.DataFrame())

        self.assertEqual(summary["market_state"], "資料不足")
        self.assertEqual(summary["abnormal_symbols"], [])
        self.assertIn("沒有足夠資料", summary["watch_points"][0])


if __name__ == "__main__":
    unittest.main()
