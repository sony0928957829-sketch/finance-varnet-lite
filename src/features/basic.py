from __future__ import annotations

import numpy as np
import pandas as pd


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate trend, return, volume and volatility features per symbol."""
    if df.empty:
        return df.copy()
    out = df.sort_values(["symbol", "datetime"]).copy()
    grouped = out.groupby("symbol", group_keys=False)

    out["return_1d"] = grouped["close"].pct_change(1)
    out["return_5d"] = grouped["close"].pct_change(5)
    out["return_20d"] = grouped["close"].pct_change(20)
    out["return_60d"] = grouped["close"].pct_change(60)

    for window in [5, 20, 60, 120, 240]:
        out[f"ma_{window}"] = grouped["close"].transform(lambda s: s.rolling(window).mean())
        out[f"ma_{window}_slope"] = grouped[f"close"].transform(lambda s, w=window: s.rolling(w).mean().pct_change(5))

    out["volume_ma20"] = grouped["volume"].transform(lambda s: s.rolling(20).mean())
    out["volume_ratio"] = out["volume"] / out["volume_ma20"].replace(0, np.nan)

    out["volatility_20"] = grouped["return_1d"].transform(lambda s: s.rolling(20).std())
    out["volatility_60"] = grouped["return_1d"].transform(lambda s: s.rolling(60).std())

    high_low = out["high"] - out["low"]
    high_prev_close = (out["high"] - grouped["close"].shift(1)).abs()
    low_prev_close = (out["low"] - grouped["close"].shift(1)).abs()
    out["true_range"] = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    out["atr_14"] = grouped["true_range"].transform(lambda s: s.rolling(14).mean())
    out["atr_pct"] = out["atr_14"] / out["close"]

    out["range_pct"] = (out["high"] - out["low"]) / out["close"]
    out["close_position"] = (out["close"] - out["low"]) / (out["high"] - out["low"]).replace(0, np.nan)

    return out
