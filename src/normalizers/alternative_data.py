from __future__ import annotations

import pandas as pd


CHIP_COLUMNS = [
    "datetime",
    "symbol",
    "market",
    "dataset",
    "metric",
    "value",
    "unit",
    "source",
    "created_at",
]

DERIVATIVE_COLUMNS = [
    "datetime",
    "symbol",
    "market",
    "dataset",
    "contract",
    "expiry",
    "option_type",
    "strike",
    "open",
    "high",
    "low",
    "close",
    "settlement",
    "volume",
    "open_interest",
    "value",
    "source",
    "created_at",
]

NEWS_COLUMNS = [
    "datetime",
    "symbol",
    "market",
    "event_type",
    "title",
    "summary",
    "url",
    "publisher",
    "source",
    "created_at",
]


def normalize_alternative_frame(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=columns)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Alternative data frame missing columns: {sorted(missing)}")

    output = frame[columns].copy()
    output["datetime"] = pd.to_datetime(output["datetime"], errors="coerce").dt.tz_localize(None)
    output = output.dropna(subset=["datetime"])
    output = output.sort_values(["datetime", "symbol"]).reset_index(drop=True)
    return output
