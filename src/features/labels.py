from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


DEFAULT_LABEL_HORIZONS = (1, 5, 10)
REQUIRED_COLUMNS = {"datetime", "symbol", "high", "low", "close"}


def range_label_columns(horizons: Iterable[int] = DEFAULT_LABEL_HORIZONS) -> list[str]:
    columns: list[str] = []
    for horizon in horizons:
        columns.extend(
            [
                f"next_{horizon}d_high_pct",
                f"next_{horizon}d_low_pct",
            ]
        )
    return columns


def add_future_range_labels(
    df: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_LABEL_HORIZONS,
) -> pd.DataFrame:
    """Add future high/low range labels for training and backtesting only.

    Each label uses trading rows after the current row within the same symbol.
    A label remains missing unless the complete future horizon is available.
    These columns must never be included in live inference features.
    """
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Label frame missing columns: {sorted(missing)}")

    normalized_horizons = tuple(dict.fromkeys(int(h) for h in horizons))
    if not normalized_horizons or any(horizon <= 0 for horizon in normalized_horizons):
        raise ValueError("Label horizons must contain positive integers.")

    out = df.sort_values(["symbol", "datetime"]).copy()
    grouped = out.groupby("symbol", group_keys=False, sort=False)

    for horizon in normalized_horizons:
        future_high = grouped["high"].transform(
            lambda series, h=horizon: series.shift(-1)
            .rolling(window=h, min_periods=h)
            .max()
            .shift(-(h - 1))
        )
        future_low = grouped["low"].transform(
            lambda series, h=horizon: series.shift(-1)
            .rolling(window=h, min_periods=h)
            .min()
            .shift(-(h - 1))
        )
        out[f"next_{horizon}d_high_pct"] = future_high / out["close"] - 1
        out[f"next_{horizon}d_low_pct"] = future_low / out["close"] - 1

    return out
