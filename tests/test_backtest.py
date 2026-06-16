from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.backtest import run_backtest, build_backtest_report


def _panel(seed: int, predictive: bool):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=400, freq="B")
    rows = []
    for sym in [f"S{i}" for i in range(12)]:
        px = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.015, len(dates))))
        fwd5 = px.shift(-5) / px - 1
        if predictive:
            sig = fwd5 + rng.normal(0, 0.02, len(dates))
        else:
            sig = pd.Series(rng.normal(0, 1, len(dates)))
        for i, d in enumerate(dates):
            rows.append({"symbol": sym, "datetime": d, "close": float(px.iloc[i]),
                         "sig": float(sig.iloc[i]) if np.isfinite(sig.iloc[i]) else np.nan})
    return pd.DataFrame(rows)


def test_predictive_beats_buy_and_hold():
    r = run_backtest(_panel(1, True), "sig", horizon=5, top_k=3, cost_bps=10)
    assert r["n_periods"] > 10
    assert r["excess_annualized_return"] > 0


def test_noise_does_not_beat_buy_and_hold():
    r = run_backtest(_panel(1, False), "sig", horizon=5, top_k=3, cost_bps=10)
    assert r["excess_annualized_return"] <= 0.05  # noise shouldn't show real edge


def test_missing_columns_degrade():
    r = run_backtest(pd.DataFrame({"symbol": ["A"], "datetime": ["2025-01-01"]}), "sig")
    assert r["n_periods"] == 0
    assert np.isnan(r["annualized_return"])


def test_report_builds():
    r = run_backtest(_panel(1, True), "sig")
    text = build_backtest_report([r], cost_bps=10, top_k=3)
    assert "策略回測" in text and "超額" in text
