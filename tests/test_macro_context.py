from __future__ import annotations

import unittest

import pandas as pd

from src.features.macro_context import add_macro_context


class MacroContextTest(unittest.TestCase):
    def test_adds_macro_risk_and_taiwan_divergence(self):
        dates = pd.date_range("2026-01-01", periods=2)
        frame = pd.DataFrame(
            [
                {"datetime": day, "symbol": symbol, "market": market, "close": close, "return_20d": return_20d}
                for day in dates
                for symbol, market, close, return_20d in [
                    ("2330.TW", "TW", 100, 0.08),
                    ("TAIEX", "TW", 20000, 0.02),
                    ("^VIX", "US_VOLATILITY", 20, 0.10),
                    ("^TNX", "US_RATE", 4.2, 0.03),
                    ("DX-Y.NYB", "US_DOLLAR", 102, 0.02),
                    ("TWD=X", "FX", 32, 0.01),
                ]
            ]
        )

        result = add_macro_context(frame)
        stock = result[result["symbol"].eq("2330.TW")].iloc[-1]

        self.assertGreater(stock["macro_risk_score"], 0)
        self.assertAlmostEqual(stock["cross_market_divergence_score"], 30.0)


if __name__ == "__main__":
    unittest.main()
