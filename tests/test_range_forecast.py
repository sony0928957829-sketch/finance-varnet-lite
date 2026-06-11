from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.range_forecast import add_range_forecasts, range_forecast_columns


class RangeForecastTest(unittest.TestCase):
    def make_prices(self, rows: int = 120) -> pd.DataFrame:
        close = 100 + np.arange(rows, dtype=float) * 0.2
        return pd.DataFrame(
            {
                "datetime": pd.date_range("2025-01-01", periods=rows, freq="D"),
                "symbol": "TEST",
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
            }
        )

    def test_forecast_columns_are_live_features_not_labels(self):
        result = add_range_forecasts(
            self.make_prices(),
            horizons=[1, 5, 10],
            window=30,
            min_periods=10,
        )

        self.assertTrue(set(range_forecast_columns()).issubset(result.columns))
        self.assertFalse(any(column.startswith("next_") for column in result.columns))
        self.assertTrue(result.iloc[-1]["pred_next_5d_high_pct"] > 0)
        self.assertTrue(result.iloc[-1]["pred_next_5d_low_pct"] < result.iloc[-1]["pred_next_5d_high_pct"])

    def test_future_changes_do_not_change_past_forecast(self):
        original = self.make_prices()
        changed = original.copy()
        changed.loc[changed.index > 90, ["high", "low", "close"]] *= 5

        first = add_range_forecasts(original, horizons=[5], window=30, min_periods=10)
        second = add_range_forecasts(changed, horizons=[5], window=30, min_periods=10)

        pd.testing.assert_series_equal(
            first.loc[:90, "pred_next_5d_high_pct"],
            second.loc[:90, "pred_next_5d_high_pct"],
        )


if __name__ == "__main__":
    unittest.main()
