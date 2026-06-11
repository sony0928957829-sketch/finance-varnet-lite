from __future__ import annotations

import pandas as pd

PRICE_COLUMNS = [
    "datetime",
    "symbol",
    "market",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
    "adjusted",
    "created_at",
]


def normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardize price data schema."""
    if df.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    missing = set(PRICE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Price frame missing columns: {sorted(missing)}")

    out = df[PRICE_COLUMNS].copy()
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.tz_localize(None)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["datetime", "symbol", "close"]).sort_values(["symbol", "datetime"])
    out = out.drop_duplicates(subset=["datetime", "symbol", "timeframe"], keep="last")
    return out.reset_index(drop=True)
