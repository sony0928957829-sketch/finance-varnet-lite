from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.normalizers.alternative_data import CHIP_COLUMNS, NEWS_COLUMNS
from src.pipeline.supplemental import collect_supplemental_data


class FakeYahoo:
    def fetch_news(self, symbols, start=None, end=None):
        return pd.DataFrame(columns=NEWS_COLUMNS)


class FakeFinMind:
    def fetch_news(self, symbols, start=None, end=None):
        return pd.DataFrame(
            [
                {
                    "datetime": "2026-06-11",
                    "symbol": "2330.TW",
                    "market": "TW",
                    "event_type": "news",
                    "title": "台灣市場測試新聞",
                    "summary": "",
                    "url": "https://example.com",
                    "publisher": "Example",
                    "source": "finmind",
                    "created_at": pd.Timestamp.now(tz="UTC"),
                }
            ],
            columns=NEWS_COLUMNS,
        )


class BrokenFinMind:
    def fetch_institutional_history(self, symbols, start=None, end=None):
        raise RuntimeError("temporary failure")


class FakeTwse:
    def fetch_institutional_history(self, symbols, start=None, end=None):
        return pd.DataFrame(
            [
                {
                    "datetime": "2026-06-11",
                    "symbol": "2330.TW",
                    "market": "TW",
                    "dataset": "institutional",
                    "metric": "foreign_net_buy",
                    "value": 100,
                    "unit": "shares",
                    "source": "twse",
                    "created_at": pd.Timestamp.now(tz="UTC"),
                }
            ],
            columns=CHIP_COLUMNS,
        )


class SupplementalRouterTest(unittest.TestCase):
    def test_news_uses_finmind_fallback_when_yahoo_is_empty(self):
        config = {
            "providers": {
                "yahoo_finance_news": {"enabled": True},
                "finmind": {"enabled": True},
            },
            "datasets": {
                "news": {
                    "market_events": {
                        "enabled": True,
                        "primary": "yahoo_finance_news",
                        "fallback": ["finmind"],
                        "history_days": 7,
                    }
                }
            },
        }
        adapters = {
            "yahoo_finance_news": FakeYahoo(),
            "finmind": FakeFinMind(),
        }

        with TemporaryDirectory() as temp_dir:
            status = collect_supplemental_data(
                config,
                symbols=["2330.TW"],
                start="2026-01-01",
                end="2026-06-11",
                output_dir=Path(temp_dir),
                provider_factory=lambda provider: adapters[provider],
            )

            route = status["news.market_events"]
            self.assertEqual(route["provider"], "finmind")
            self.assertTrue(route["fallback_used"])
            self.assertTrue((Path(temp_dir) / "news.parquet").exists())

    def test_chip_uses_twse_fallback_when_finmind_errors(self):
        config = {
            "providers": {
                "finmind": {"enabled": True},
                "twse": {"enabled": True},
            },
            "datasets": {
                "chip": {
                    "taiwan_institutional": {
                        "enabled": True,
                        "primary": "finmind",
                        "fallback": ["twse"],
                        "history_days": 7,
                    }
                }
            },
        }
        adapters = {
            "finmind": BrokenFinMind(),
            "twse": FakeTwse(),
        }

        with TemporaryDirectory() as temp_dir:
            status = collect_supplemental_data(
                config,
                symbols=["2330.TW"],
                start="2026-01-01",
                end="2026-06-11",
                output_dir=Path(temp_dir),
                provider_factory=lambda provider: adapters[provider],
            )

            route = status["chip.taiwan_institutional"]
            self.assertEqual(route["provider"], "twse")
            self.assertTrue(route["fallback_used"])
            self.assertEqual(route["attempts"][0]["status"], "error")
            self.assertTrue((Path(temp_dir) / "chip.parquet").exists())


if __name__ == "__main__":
    unittest.main()
