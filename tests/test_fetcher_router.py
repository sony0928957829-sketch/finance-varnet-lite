from __future__ import annotations

import unittest

import pandas as pd

from src.fetchers.router import fetch_prices_with_fallback
from src.fetchers.yfinance_fetcher import YFINANCE_COLUMNS


class FakeFetcher:
    def __init__(self, provider):
        self.provider = provider

    def fetch_price_history(self, symbols, **kwargs):
        available = {
            "primary": {"A"},
            "fallback": {"B"},
        }[self.provider]
        rows = []
        for symbol in symbols:
            if symbol not in available:
                continue
            rows.append(
                {
                    "datetime": pd.Timestamp("2026-06-01"),
                    "symbol": symbol,
                    "market": "TEST",
                    "timeframe": "1d",
                    "open": 1,
                    "high": 1,
                    "low": 1,
                    "close": 1,
                    "volume": 1,
                    "source": self.provider,
                    "adjusted": True,
                    "created_at": pd.Timestamp.now(tz="UTC"),
                }
            )
        return pd.DataFrame(rows, columns=YFINANCE_COLUMNS)


class FetcherRouterTest(unittest.TestCase):
    def test_missing_symbol_uses_ordered_fallback(self):
        config = {
            "providers": {
                "primary": {"enabled": True},
                "fallback": {"enabled": True},
            },
            "datasets": {
                "prices": {
                    "test": {
                        "enabled": True,
                        "primary": "primary",
                        "fallback": ["fallback"],
                        "symbols": ["A", "B"],
                    }
                }
            },
        }

        frame, status = fetch_prices_with_fallback(
            config,
            primary_provider="primary",
            start="2026-01-01",
            end="2026-06-01",
            fetcher_factory=lambda provider, **kwargs: FakeFetcher(provider),
        )

        self.assertEqual(frame["symbol"].tolist(), ["A", "B"])
        self.assertEqual(frame["source"].tolist(), ["primary", "fallback"])
        self.assertEqual(status["prices.test"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
