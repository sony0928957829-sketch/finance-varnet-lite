from __future__ import annotations

import unittest

import pandas as pd

from src.fetchers.finmind_fetcher import FinMindFetcher
from src.fetchers.yfinance_fetcher import YFINANCE_COLUMNS
from src.normalizers.alternative_data import CHIP_COLUMNS


class FakeFinMindFetcher(FinMindFetcher):
    def fetch_dataset(self, dataset, **kwargs):
        if dataset == "TaiwanStockPriceAdj":
            return pd.DataFrame(
                [
                    {
                        "date": "2026-06-01",
                        "stock_id": "2330",
                        "open": 1000,
                        "max": 1020,
                        "min": 990,
                        "close": 1010,
                        "Trading_Volume": 12345,
                    }
                ]
            )
        if dataset == "TaiwanExchangeRate":
            return pd.DataFrame(
                [
                    {
                        "date": "2026-06-01",
                        "currency": "USD",
                        "spot_buy": 31.90,
                        "spot_sell": 32.10,
                    }
                ]
            )
        if dataset == "TaiwanStockInstitutionalInvestorsBuySell":
            return pd.DataFrame(
                [
                    {
                        "date": "2026-06-01",
                        "name": "Foreign_Investor",
                        "buy": 100,
                        "sell": 40,
                    }
                ]
            )
        if dataset == "TaiwanStockMarginPurchaseShortSale":
            return pd.DataFrame(
                [
                    {
                        "date": "2026-06-01",
                        "MarginPurchaseTodayBalance": 500,
                        "ShortSaleTodayBalance": 20,
                    }
                ]
            )
        return pd.DataFrame()


class FinMindFetcherTest(unittest.TestCase):
    def test_normalizes_taiwan_stock_price(self):
        result = FakeFinMindFetcher().fetch_price_history(
            ["2330.TW"],
            start="2026-06-01",
            end="2026-06-02",
        )

        self.assertEqual(result.columns.tolist(), YFINANCE_COLUMNS)
        self.assertEqual(result.loc[0, "symbol"], "2330.TW")
        self.assertEqual(result.loc[0, "market"], "TW")
        self.assertEqual(result.loc[0, "close"], 1010)

    def test_normalizes_institutional_and_margin_data(self):
        result = pd.concat(
            [
                FakeFinMindFetcher().fetch_institutional_history(
                    ["2330.TW"],
                    start="2026-06-01",
                    end="2026-06-02",
                ),
                FakeFinMindFetcher().fetch_margin_short_history(
                    ["2330.TW"],
                    start="2026-06-01",
                    end="2026-06-02",
                ),
            ],
            ignore_index=True,
        )

        self.assertEqual(result.columns.tolist(), CHIP_COLUMNS)
        values = dict(zip(result["metric"], result["value"]))
        self.assertEqual(values["foreign_net_buy"], 60)
        self.assertEqual(values["margin_balance"], 500)
        self.assertEqual(values["short_balance"], 20)

    def test_normalizes_usd_twd_exchange_rate(self):
        result = FakeFinMindFetcher().fetch_price_history(
            ["TWD=X"],
            start="2026-06-01",
            end="2026-06-02",
        )

        self.assertEqual(result.loc[0, "market"], "FX")
        self.assertEqual(result.loc[0, "close"], 32.0)
        self.assertEqual(result.loc[0, "source"], "finmind")


if __name__ == "__main__":
    unittest.main()
