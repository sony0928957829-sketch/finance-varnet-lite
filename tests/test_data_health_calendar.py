from __future__ import annotations

from datetime import date

import pandas as pd

from src.health.data_health import evaluate_price_health

CONFIG = {
    "required_columns": [
        "datetime", "symbol", "market", "timeframe",
        "open", "high", "low", "close", "volume", "source",
    ],
    "minimum_rows_per_symbol": 2,
    "freshness_max_age_days": {"TW": 4, "default": 4},
    "long_gap_days": {"TW": 4, "default": 4},
    "fail_on": ["stale_data", "long_gap"],
}


def _bar(day, symbol, close=100.0):
    return {
        "datetime": day, "symbol": symbol, "market": "TW", "timeframe": "1d",
        "open": close - 1, "high": close + 2, "low": close - 2,
        "close": close, "volume": 1000, "source": "yfinance",
    }


def _tw_frame_with_shared_holiday(extra_hole_symbol: str | None = None) -> pd.DataFrame:
    # Three TW symbols. The whole market is closed 2026-02-16..2026-02-20 (a
    # CNY-like 9-day closure incl. weekends): NO symbol has bars on those days.
    sessions = (
        pd.bdate_range("2026-02-02", "2026-02-13").tolist()  # before closure
        + pd.bdate_range("2026-02-23", "2026-03-06").tolist()  # after closure
    )
    symbols = ["2330.TW", "2317.TW", "2454.TW"]
    rows = []
    for sym in symbols:
        for d in sessions:
            # Optionally punch a *real* hole for one symbol (market traded, it didn't).
            if extra_hole_symbol == sym and d in pd.bdate_range("2026-03-02", "2026-03-06"):
                continue
            rows.append(_bar(d.strftime("%Y-%m-%d"), sym))
    return pd.DataFrame(rows)


def test_market_wide_holiday_gap_is_not_flagged():
    frame = _tw_frame_with_shared_holiday()
    report = evaluate_price_health(
        frame,
        expected_symbols=["2330.TW", "2317.TW", "2454.TW"],
        as_of=date(2026, 3, 6),
        primary_source="yfinance",
        config=CONFIG,
    )
    codes = {i["code"] for i in report["issues"]}
    assert "long_gap" not in codes, report["issues"]
    assert "stale_data" not in codes, report["issues"]
    assert report["status"] == "healthy"


def test_real_per_symbol_hole_is_still_flagged():
    # One symbol skips a stretch the rest of the market traded -> real hole.
    frame = _tw_frame_with_shared_holiday(extra_hole_symbol="2454.TW")
    report = evaluate_price_health(
        frame,
        expected_symbols=["2330.TW", "2317.TW", "2454.TW"],
        as_of=date(2026, 3, 6),
        primary_source="yfinance",
        config=CONFIG,
    )
    gap_issues = [i for i in report["issues"] if i["code"] in ("long_gap", "stale_data")]
    assert gap_issues, "a genuine per-symbol hole should still be flagged"
    assert all(i.get("symbol") == "2454.TW" for i in gap_issues), gap_issues
