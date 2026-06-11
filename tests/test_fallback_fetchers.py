from __future__ import annotations

from datetime import date
import unittest

import pandas as pd

from src.fetchers.fred_fetcher import FredFetcher
from src.fetchers.twse_fetcher import TwseFetcher


class FallbackFetcherTest(unittest.TestCase):
    def test_twse_snapshot_normalizes_latest_bar(self):
        raw = pd.DataFrame(
            [
                {
                    "Code": "2330",
                    "OpeningPrice": "1,000",
                    "HighestPrice": "1,020",
                    "LowestPrice": "990",
                    "ClosingPrice": "1,010",
                    "TradeVolume": "12,345",
                }
            ]
        )

        result = TwseFetcher().normalize_snapshot(
            raw,
            ["2330.TW"],
            as_of=date(2026, 6, 1),
        )

        self.assertEqual(result.loc[0, "symbol"], "2330.TW")
        self.assertEqual(result.loc[0, "close"], 1010)
        self.assertEqual(result.loc[0, "source"], "twse")

    def test_fred_series_normalizes_yield_as_price_like_observation(self):
        raw = pd.DataFrame(
            {
                "DATE": ["2026-06-01", "2026-06-02"],
                "DGS10": ["4.20", "."],
            }
        )

        result = FredFetcher().normalize_series(raw, symbol="^TNX", series="DGS10")

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "close"], 4.2)
        self.assertEqual(result.loc[0, "market"], "US_RATE")

    def test_fred_vix_uses_official_vix_series(self):
        raw = pd.DataFrame(
            {
                "observation_date": ["2026-06-10"],
                "VIXCLS": ["22.22"],
            }
        )

        result = FredFetcher().normalize_series(
            raw,
            symbol="^VIX",
            series="VIXCLS",
        )

        self.assertEqual(result.loc[0, "close"], 22.22)
        self.assertEqual(result.loc[0, "market"], "US_VOLATILITY")

    def test_twse_normalizes_institutional_and_margin_snapshots(self):
        fetcher = TwseFetcher()
        institutional = pd.DataFrame(
            [
                {
                    "證券代號": "2330",
                    "外陸資買進股數(不含外資自營商)": "1,000",
                    "外陸資賣出股數(不含外資自營商)": "400",
                    "投信買進股數": "200",
                    "投信賣出股數": "50",
                    "自營商買賣超股數": "-20",
                }
            ]
        )
        margin = pd.DataFrame(
            [
                {
                    "股票代號": "2330",
                    "融資今日餘額": "5,000",
                    "融券今日餘額": "300",
                }
            ]
        )

        institution_result = fetcher.normalize_institutional_snapshot(
            institutional,
            ["2330.TW"],
            as_of=date(2026, 6, 11),
        )
        margin_result = fetcher.normalize_margin_snapshot(
            margin,
            ["2330.TW"],
            as_of=date(2026, 6, 11),
        )

        institution_values = dict(
            zip(institution_result["metric"], institution_result["value"])
        )
        margin_values = dict(zip(margin_result["metric"], margin_result["value"]))
        self.assertEqual(institution_values["foreign_net_buy"], 600)
        self.assertEqual(institution_values["investment_trust_net_buy"], 150)
        self.assertEqual(institution_values["dealer_net_buy"], -20)
        self.assertEqual(margin_values["margin_balance"], 5000)
        self.assertEqual(margin_values["short_balance"], 300)


if __name__ == "__main__":
    unittest.main()
