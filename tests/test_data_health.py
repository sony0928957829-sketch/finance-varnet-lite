from __future__ import annotations

from datetime import date
import unittest

import pandas as pd

from src.health.data_health import (
    DataHealthError,
    evaluate_price_health,
    raise_for_health_errors,
)


def healthy_frame() -> pd.DataFrame:
    rows: list[dict] = []
    for symbol, market in [("NVDA", "US"), ("BTC-USD", "CRYPTO")]:
        for day, close in [("2026-06-09", 100.0), ("2026-06-10", 101.0)]:
            rows.append(
                {
                    "datetime": day,
                    "symbol": symbol,
                    "market": market,
                    "timeframe": "1d",
                    "open": close - 1,
                    "high": close + 2,
                    "low": close - 2,
                    "close": close,
                    "volume": 1000,
                    "source": "yfinance",
                }
            )
    return pd.DataFrame(rows)


HEALTH_CONFIG = {
    "required_columns": [
        "datetime",
        "symbol",
        "market",
        "timeframe",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
    ],
    "minimum_rows_per_symbol": 2,
    "freshness_max_age_days": {"US": 4, "CRYPTO": 2, "default": 4},
    "long_gap_days": {"US": 4, "CRYPTO": 2, "default": 4},
    "fail_on": [
        "empty_data",
        "missing_columns",
        "missing_symbol",
        "insufficient_rows",
        "invalid_datetime",
        "null_price",
        "non_positive_price",
        "invalid_ohlc",
        "duplicate_bar",
        "stale_data",
    ],
}


class DataHealthTest(unittest.TestCase):
    def test_healthy_data_passes(self):
        report = evaluate_price_health(
            healthy_frame(),
            expected_symbols=["NVDA", "BTC-USD"],
            as_of=date(2026, 6, 11),
            primary_source="yfinance",
            config=HEALTH_CONFIG,
        )

        self.assertEqual(report["status"], "healthy")
        self.assertEqual(report["summary"]["error_count"], 0)
        raise_for_health_errors(report)

    def test_missing_symbol_and_invalid_prices_fail(self):
        frame = healthy_frame()
        frame = frame[frame["symbol"].eq("NVDA")].copy()
        frame.loc[frame.index[0], "close"] = 0

        report = evaluate_price_health(
            frame,
            expected_symbols=["NVDA", "BTC-USD"],
            as_of=date(2026, 6, 11),
            primary_source="yfinance",
            config=HEALTH_CONFIG,
        )

        codes = {issue["code"] for issue in report["issues"]}
        self.assertEqual(report["status"], "error")
        self.assertIn("missing_symbol", codes)
        self.assertIn("non_positive_price", codes)
        with self.assertRaises(DataHealthError):
            raise_for_health_errors(report)

    def test_stale_data_fails_and_fallback_is_reported(self):
        frame = healthy_frame()
        frame["source"] = "backup"

        report = evaluate_price_health(
            frame,
            expected_symbols=["NVDA", "BTC-USD"],
            as_of=date(2026, 6, 20),
            primary_source="yfinance",
            config=HEALTH_CONFIG,
        )

        codes = {issue["code"] for issue in report["issues"]}
        self.assertEqual(report["status"], "error")
        self.assertIn("stale_data", codes)
        self.assertIn("fallback_source", codes)

    def test_health_error_names_the_failing_symbol(self):
        frame = healthy_frame()
        nvda_index = frame.index[frame["symbol"].eq("NVDA")][0]
        frame.loc[nvda_index, "high"] = 1
        report = evaluate_price_health(
            frame,
            expected_symbols=["NVDA", "BTC-USD"],
            as_of=date(2026, 6, 11),
            primary_source="yfinance",
            config=HEALTH_CONFIG,
        )

        with self.assertRaisesRegex(DataHealthError, "NVDA: .*OHLC"):
            raise_for_health_errors(report)


if __name__ == "__main__":
    unittest.main()
