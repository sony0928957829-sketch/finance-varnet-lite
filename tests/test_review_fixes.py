from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.macro_context import add_macro_context
from src.evaluation.range_coverage import evaluate_range_coverage
from src.scoring.scores import _trend_score, _momentum_score, add_scores


def _macro_frame() -> pd.DataFrame:
    dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
    rows = []
    for i, dt in enumerate(dates):
        rows.append(
            {"datetime": dt, "symbol": "2330.TW", "market": "TW",
             "close": 1000 + i, "return_20d": 0.01}
        )
        rows.append(
            {"datetime": dt, "symbol": "NVDA", "market": "US",
             "close": 140 + i, "return_20d": 0.02}
        )
        rows.append(
            {"datetime": dt, "symbol": "^VIX", "market": "US",
             "close": [15.0, 40.0, 15.0][i], "return_20d": 0.0}
        )
    return pd.DataFrame(rows)


def test_tw_macro_risk_is_lagged_one_day():
    """A VIX spike on day 2 must hit TW rows on day 3, US rows on day 2."""
    out = add_macro_context(_macro_frame())
    day2 = pd.Timestamp("2026-01-06")
    day3 = pd.Timestamp("2026-01-07")

    tw_day2 = out.loc[(out["symbol"] == "2330.TW") & (out["datetime"] == day2), "macro_risk_score"].iloc[0]
    tw_day3 = out.loc[(out["symbol"] == "2330.TW") & (out["datetime"] == day3), "macro_risk_score"].iloc[0]
    us_day2 = out.loc[(out["symbol"] == "NVDA") & (out["datetime"] == day2), "macro_risk_score"].iloc[0]

    # US sees the spike same-date; TW must not (its close happened first).
    assert us_day2 > tw_day2
    # TW sees the spike the following date.
    assert tw_day3 > tw_day2


def test_coverage_metrics_basic():
    n = 10
    frame = pd.DataFrame(
        {
            "symbol": ["X"] * n,
            "pred_next_1d_low_pct": [-0.02] * n,
            "pred_next_1d_high_pct": [0.02] * n,
            "next_1d_low_pct": [-0.01] * 8 + [-0.05] * 2,
            "next_1d_high_pct": [0.01] * n,
        }
    )
    result = evaluate_range_coverage(frame, horizons=[1])
    row = result.iloc[0]
    assert row["n_observations"] == n
    assert row["coverage"] == pytest.approx(0.8)
    assert row["low_side_coverage"] == pytest.approx(0.8)
    assert row["high_side_coverage"] == pytest.approx(1.0)
    assert row["mean_band_width"] == pytest.approx(0.04)


def test_trend_score_nan_during_warmup():
    assert np.isnan(_trend_score(pd.Series({"close": 100.0})))


def test_momentum_single_day_not_triple_counted():
    # One +3% day, flat otherwise: r1 = r5 = r20 = 3%.
    row = pd.Series({"return_1d": 0.03, "return_5d": 0.03, "return_20d": 0.03})
    score = _momentum_score(row)
    # Old overlapping formula gave 50 + 0.03*(500+250+100) = 75.5.
    assert score == pytest.approx(65.0)


def test_index_volume_score_suppressed():
    frame = pd.DataFrame(
        {
            "symbol": ["TAIEX", "2330.TW"],
            "close": [20000.0, 1000.0],
            "volume_ratio": [1.0, 1.0],
        }
    )
    out = add_scores(frame)
    assert np.isnan(out.loc[out["symbol"] == "TAIEX", "volume_score"].iloc[0])
    assert np.isfinite(out.loc[out["symbol"] == "2330.TW", "volume_score"].iloc[0])
