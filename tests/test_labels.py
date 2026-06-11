from __future__ import annotations

import unittest

import pandas as pd
from pandas.testing import assert_series_equal

from src.features.labels import add_future_range_labels, range_label_columns


def price_frame(symbol: str, start: str, periods: int) -> pd.DataFrame:
    sequence = pd.Series(range(periods), dtype=float)
    return pd.DataFrame(
        {
            "datetime": pd.date_range(start, periods=periods, freq="D"),
            "symbol": symbol,
            "high": 110.0 + sequence * 10.0,
            "low": 90.0 - sequence * 2.0,
            "close": 100.0 + sequence,
        }
    )


class FutureRangeLabelsTest(unittest.TestCase):
    def test_calculates_exact_1d_5d_and_10d_ranges(self):
        frame = price_frame("AAA", "2026-01-01", 12)
        frame.loc[0, ["high", "low"]] = [9999.0, -9999.0]

        labeled = add_future_range_labels(frame)
        first = labeled.iloc[0]

        self.assertAlmostEqual(first["next_1d_high_pct"], 120.0 / 100.0 - 1)
        self.assertAlmostEqual(first["next_1d_low_pct"], 88.0 / 100.0 - 1)
        self.assertAlmostEqual(first["next_5d_high_pct"], 160.0 / 100.0 - 1)
        self.assertAlmostEqual(first["next_5d_low_pct"], 80.0 / 100.0 - 1)
        self.assertAlmostEqual(first["next_10d_high_pct"], 210.0 / 100.0 - 1)
        self.assertAlmostEqual(first["next_10d_low_pct"], 70.0 / 100.0 - 1)

    def test_current_bar_does_not_change_its_future_labels(self):
        original = price_frame("AAA", "2026-01-01", 12)
        changed = original.copy()
        changed.loc[0, ["high", "low"]] = [1_000_000.0, -1_000_000.0]

        original_labels = add_future_range_labels(original)
        changed_labels = add_future_range_labels(changed)
        columns = range_label_columns()

        assert_series_equal(
            original_labels.loc[0, columns],
            changed_labels.loc[0, columns],
            check_names=False,
        )

    def test_incomplete_horizons_stay_missing_and_do_not_cross_symbols(self):
        frame = pd.concat(
            [
                price_frame("AAA", "2026-01-01", 12),
                price_frame("BBB", "2026-02-01", 12),
            ],
            ignore_index=True,
        )
        frame.loc[frame["symbol"].eq("BBB"), "high"] += 10_000.0
        labeled = add_future_range_labels(frame)

        for symbol in ["AAA", "BBB"]:
            symbol_rows = labeled[labeled["symbol"].eq(symbol)].reset_index(drop=True)
            self.assertTrue(symbol_rows.tail(1)["next_1d_high_pct"].isna().all())
            self.assertTrue(symbol_rows.tail(5)["next_5d_high_pct"].isna().all())
            self.assertTrue(symbol_rows.tail(10)["next_10d_high_pct"].isna().all())
            self.assertTrue(symbol_rows.tail(10)["next_10d_low_pct"].isna().all())

        aaa_first = labeled[labeled["symbol"].eq("AAA")].iloc[0]
        self.assertAlmostEqual(aaa_first["next_10d_high_pct"], 210.0 / 100.0 - 1)

    def test_rejects_invalid_horizons(self):
        frame = price_frame("AAA", "2026-01-01", 12)

        with self.assertRaises(ValueError):
            add_future_range_labels(frame, horizons=[0, 5])


if __name__ == "__main__":
    unittest.main()
