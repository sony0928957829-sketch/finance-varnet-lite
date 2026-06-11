from __future__ import annotations

import pandas as pd

from .labels import add_future_range_labels


def add_future_range_targets(df: pd.DataFrame, horizons: list[int] | None = None) -> pd.DataFrame:
    """Backward-compatible alias for the training label generator."""
    return add_future_range_labels(df, horizons=horizons or [1, 5, 10])
