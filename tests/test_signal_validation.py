from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.signal_validation import (
    add_forward_returns,
    evaluate_signals,
    update_track_record,
)


def _make_frame(seed: int = 0, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    rows = []
    for sym in ("A", "B"):
        drift = rng.normal(0, 0.01, n)
        px, series = 100.0, []
        for i in range(n):
            px *= 1 + drift[i]
            series.append(px)
        close = pd.Series(series)
        fwd5 = close.shift(-5) / close - 1
        good = (50 + np.clip(fwd5 * 1000, -49, 49)).fillna(50)
        noise = rng.uniform(0, 100, n)
        for i, d in enumerate(dates):
            rows.append(dict(symbol=sym, datetime=d, close=series[i],
                             good_score=good.iloc[i], noise_score=noise[i]))
    return pd.DataFrame(rows)


def test_forward_returns_last_h_rows_are_nan():
    df = _make_frame()
    out = add_forward_returns(df, horizons=(5,))
    for _, g in out.groupby("symbol"):
        g = g.sort_values("datetime")
        assert g["fwd_return_5d"].tail(5).isna().all()
        assert g["fwd_return_5d"].head(-5).notna().all()


def test_predictive_signal_beats_random_and_noise_does_not():
    df = _make_frame()
    sc = evaluate_signals(df, ["good_score", "noise_score"], horizons=(5,))
    good = sc[sc["signal"] == "good_score"].iloc[0]
    noise = sc[sc["signal"] == "noise_score"].iloc[0]
    # A signal built from the future must show strong, significant edge.
    assert good["direction_hit_rate"] > 0.9
    assert good["ic"] > 0.9
    assert abs(good["ic_tstat"]) > 2
    # Pure noise must look like a coin flip with ~zero IC.
    assert abs(noise["direction_hit_rate"] - 0.5) < 0.1
    assert abs(noise["ic"]) < 0.2
    assert abs(noise["hit_zstat"]) < 2


def test_missing_signal_is_reported_not_crashed():
    df = _make_frame()
    sc = evaluate_signals(df, ["does_not_exist"], horizons=(5,))
    row = sc.iloc[0]
    assert row["n"] == 0
    assert np.isnan(row["ic"])


def test_too_few_observations_degrade_to_nan():
    df = _make_frame(n=10)  # fewer rows than min_obs after losing last 5
    sc = evaluate_signals(df, ["good_score"], horizons=(5,))
    assert np.isnan(sc.iloc[0]["ic"])


def test_tz_aware_datetime_does_not_crash():
    df = _make_frame()
    df["datetime"] = df["datetime"].dt.tz_localize("Asia/Taipei")
    sc = evaluate_signals(df, ["good_score"], horizons=(5,))
    assert int(sc.iloc[0]["n"]) > 0


def test_track_record_is_idempotent_per_day(tmp_path):
    df = _make_frame()
    sc = evaluate_signals(df, ["good_score"], horizons=(5,))
    p = tmp_path / "tr.parquet"
    update_track_record(sc, p, as_of="2025-06-13")
    update_track_record(sc, p, as_of="2025-06-13")  # same day replaces
    h = update_track_record(sc, p, as_of="2025-06-14")
    assert sorted(h["as_of"].dt.date.astype(str).unique()) == ["2025-06-13", "2025-06-14"]
    assert len(h) == 2 * len(sc)


def test_neutral_none_skips_direction_test():
    df = _make_frame()
    sc = evaluate_signals(df, ["good_score"], horizons=(5,), neutral=None)
    assert np.isnan(sc.iloc[0]["direction_hit_rate"])
    assert sc.iloc[0]["direction_n"] == 0
    assert np.isfinite(sc.iloc[0]["ic"])  # IC still computed
