from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.factor_standardize import add_standardized_factors
from src.models.ml_forecast import add_ml_forecast, ml_forecast_column


def _panel(seed: int, predictive: bool):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    syms = [f"S{i}" for i in range(8)]
    rows = []
    for sym in syms:
        px = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.012, len(dates))))
        fwd5 = px.shift(-5) / px - 1
        if predictive:
            f = fwd5 + rng.normal(0, 0.02, len(dates))  # future-linked signal + noise
        else:
            f = pd.Series(rng.normal(0, 1, len(dates)))  # pure noise
        for i, d in enumerate(dates):
            rows.append({"datetime": d, "symbol": sym, "close": float(px.iloc[i]),
                         "f": float(f.iloc[i]) if np.isfinite(f.iloc[i]) else np.nan})
    return pd.DataFrame(rows)


def _oos_ic(df):
    df = add_standardized_factors(df, ["f"])
    out = add_ml_forecast(df, ["f_cs"], horizon=5, min_train_dates=60,
                          retrain_every=21, min_train_rows=40)
    col = ml_forecast_column(5)
    out = out.sort_values(["symbol", "datetime"])
    out["real"] = out.groupby("symbol")["close"].transform(lambda s: s.shift(-5) / s - 1)
    mask = out[col].notna() & out["real"].notna()
    sub = out[mask]
    return float(sub[col].rank().corr(sub["real"].rank())), int(mask.sum())


def _median_ic(predictive, seeds=range(5)):
    ics = [_oos_ic(_panel(s, predictive=predictive))[0] for s in seeds]
    return float(np.median(ics))


def test_learns_real_signal_out_of_sample():
    # Median across seeds is stable; overlapping horizons make a single-seed IC noisy.
    assert _median_ic(predictive=True) > 0.3


def test_rejects_pure_noise_and_signal_dominates():
    noise = abs(_median_ic(predictive=False))
    signal = _median_ic(predictive=True)
    assert noise < 0.1, f"median |noise IC| should be ~0, got {noise}"
    assert signal > 3 * noise, "real signal must clearly dominate noise"


def test_insufficient_history_degrades_to_nan():
    df = pd.DataFrame({
        "datetime": pd.date_range("2025-01-01", periods=10, freq="B").tolist() * 1,
        "symbol": ["A"] * 10,
        "close": np.linspace(100, 110, 10),
        "f_cs": np.linspace(-1, 1, 10),
    })
    out = add_ml_forecast(df, ["f_cs"], horizon=5, min_train_dates=120)
    assert out[ml_forecast_column(5)].isna().all()


def test_missing_features_safe():
    df = pd.DataFrame({"datetime": pd.date_range("2025-01-01", periods=5),
                       "symbol": ["A"] * 5, "close": [1.0, 2, 3, 4, 5]})
    out = add_ml_forecast(df, ["nope_cs"], horizon=5)
    assert ml_forecast_column(5) in out.columns
    assert out[ml_forecast_column(5)].isna().all()
