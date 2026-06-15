from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.chip_flow import add_chip_flow_features


def test_post_close_taiwan_data_starts_next_business_day(tmp_path: Path):
    dates = pd.date_range("2026-01-01", periods=30, freq="B")
    prices = pd.DataFrame(
        [
            {
                "datetime": day,
                "symbol": symbol,
                "market": market,
                "close": 100.0,
            }
            for day in dates
            for symbol, market in (("2330.TW", "TW"), ("NVDA", "US"))
        ]
    )
    chip = pd.DataFrame(
        {
            "datetime": dates,
            "symbol": "2330.TW",
            "dataset": "institutional",
            "metric": "foreign_net_buy",
            "value": np.arange(1, len(dates) + 1, dtype=float),
            "unit": "shares",
        }
    )
    options = pd.DataFrame(
        [
            {
                "datetime": day,
                "dataset": "taiwan_options",
                "option_type": option_type,
                "volume": volume,
            }
            for index, day in enumerate(dates)
            for option_type, volume in (
                ("put", 100 + index),
                ("call", 200),
            )
        ]
    )
    chip.to_parquet(
        tmp_path / "chip_taiwan_institutional.parquet",
        index=False,
    )
    options.to_parquet(
        tmp_path / "derivatives_taiwan_options.parquet",
        index=False,
    )

    result = add_chip_flow_features(prices, alt_dir=tmp_path)
    taiwan = result.loc[result["symbol"].eq("2330.TW")].reset_index(drop=True)
    us = result.loc[result["symbol"].eq("NVDA")].reset_index(drop=True)

    assert pd.isna(taiwan.loc[19, "institutional_flow_score"])
    assert pd.notna(taiwan.loc[20, "institutional_flow_score"])
    assert pd.isna(taiwan.loc[19, "options_sentiment_score"])
    assert pd.notna(taiwan.loc[20, "options_sentiment_score"])
    assert us["put_call_ratio"].isna().all()
    assert us["options_sentiment_score"].isna().all()


def test_future_alternative_rows_do_not_change_past_features(tmp_path: Path):
    dates = pd.date_range("2026-01-01", periods=30, freq="B")
    prices = pd.DataFrame(
        {
            "datetime": dates,
            "symbol": "2330.TW",
            "market": "TW",
            "close": 100.0,
        }
    )
    base = pd.DataFrame(
        {
            "datetime": dates,
            "symbol": "2330.TW",
            "dataset": "institutional",
            "metric": "foreign_net_buy",
            "value": np.arange(1, len(dates) + 1, dtype=float),
            "unit": "shares",
        }
    )
    path = tmp_path / "chip_taiwan_institutional.parquet"
    base.to_parquet(path, index=False)
    original = add_chip_flow_features(prices, alt_dir=tmp_path)

    changed = base.copy()
    changed.loc[changed.index[-1], "value"] = 1_000_000.0
    changed.to_parquet(path, index=False)
    rerun = add_chip_flow_features(prices, alt_dir=tmp_path)

    pd.testing.assert_series_equal(
        original["institutional_flow_score"].iloc[:-1].reset_index(drop=True),
        rerun["institutional_flow_score"].iloc[:-1].reset_index(drop=True),
        check_names=False,
    )
