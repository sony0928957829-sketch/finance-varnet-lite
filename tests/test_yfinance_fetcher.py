from __future__ import annotations

from datetime import date
import sys
import types
import unittest
from unittest.mock import patch

import pandas as pd

from src.fetchers.yfinance_fetcher import (
    YFINANCE_COLUMNS,
    YFinanceFetcher,
)


class YFinanceFetcherTest(unittest.TestCase):
    def test_flattens_single_ticker_multiindex_columns(self):
        index = pd.DatetimeIndex(["2026-06-01"], name="Date")
        columns = pd.MultiIndex.from_product(
            [["Close", "High", "Low", "Open", "Volume"], ["NVDA"]],
            names=["Price", "Ticker"],
        )
        downloaded = pd.DataFrame(
            [[100.0, 102.0, 99.0, 101.0, 12345.0]],
            index=index,
            columns=columns,
        )
        fake_yfinance = types.SimpleNamespace(download=lambda *args, **kwargs: downloaded)

        with patch.dict(sys.modules, {"yfinance": fake_yfinance}):
            result = YFinanceFetcher().fetch_price_history(
                ["NVDA"],
                start=date(2026, 6, 1),
                end=date(2026, 6, 2),
            )

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "symbol"], "NVDA")
        self.assertEqual(result.loc[0, "close"], 100.0)
        self.assertEqual(result.loc[0, "volume"], 12345.0)
        self.assertEqual(result.columns.tolist(), YFINANCE_COLUMNS)

    def test_supports_priority_us_and_crypto_symbols(self):
        requested: list[str] = []

        def fake_download(symbol, **kwargs):
            requested.append(symbol)
            index = pd.DatetimeIndex(["2026-06-01"], name="Date")
            columns = pd.MultiIndex.from_product(
                [["Close", "High", "Low", "Open", "Volume"], [symbol]],
                names=["Price", "Ticker"],
            )
            return pd.DataFrame(
                [[100.0, 102.0, 99.0, 101.0, 12345.0]],
                index=index,
                columns=columns,
            )

        fake_yfinance = types.SimpleNamespace(download=fake_download)
        priority_symbols = ["NVDA", "TSLA", "AMD", "BTC-USD"]

        with patch.dict(sys.modules, {"yfinance": fake_yfinance}):
            result = YFinanceFetcher().fetch_price_history(
                priority_symbols,
                start=date(2026, 6, 1),
                end=date(2026, 6, 2),
            )

        self.assertEqual(requested, priority_symbols)
        self.assertEqual(result["symbol"].tolist(), priority_symbols)
        self.assertEqual(result["market"].tolist(), ["US", "US", "US", "CRYPTO"])
        self.assertTrue((result["source"] == "yfinance").all())
        self.assertEqual(result.columns.tolist(), YFINANCE_COLUMNS)

    def test_empty_symbol_list_returns_standard_schema(self):
        fake_yfinance = types.SimpleNamespace(download=lambda *args, **kwargs: None)

        with patch.dict(sys.modules, {"yfinance": fake_yfinance}):
            result = YFinanceFetcher().fetch_price_history(
                [],
                start=date(2026, 6, 1),
                end=date(2026, 6, 2),
            )

        self.assertTrue(result.empty)
        self.assertEqual(result.columns.tolist(), YFINANCE_COLUMNS)


if __name__ == "__main__":
    unittest.main()
