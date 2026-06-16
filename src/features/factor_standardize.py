from __future__ import annotations

"""Cross-sectional factor standardization.

The scoring layer historically combined hand-scaled signals with fixed weights
(a guessed linear model). To let a model learn the weights from data instead,
factors must first be put on a comparable, regime-robust scale.

This module standardizes each factor *cross-sectionally* -- per datetime,
across symbols -- so the output answers "how does this stock rank against its
peers today", which is what most equity cross-sectional models consume.

Two methods:
- "zscore": (x - mean) / std across symbols on that date (default).
- "rank":   percentile rank in [-0.5, 0.5], robust to outliers/fat tails.

A date needs at least `min_symbols` non-NaN values or the whole date degrades
to NaN for that factor (too few peers to standardize against). Missing input
columns are skipped quietly rather than raising.
"""

import numpy as np
import pandas as pd

DEFAULT_SUFFIX = "_cs"


def add_standardized_factors(
    df: pd.DataFrame,
    factor_columns: list[str] | tuple[str, ...],
    *,
    method: str = "zscore",
    suffix: str = DEFAULT_SUFFIX,
    min_symbols: int = 3,
    group_columns: tuple[str, ...] = ("datetime",),
) -> pd.DataFrame:
    """Add cross-sectionally standardized versions of `factor_columns`.

    Output columns are named f"{col}{suffix}". Returns a copy; never raises on
    missing columns or thin cross-sections (those become NaN).
    """
    if method not in ("zscore", "rank"):
        raise ValueError("method must be 'zscore' or 'rank'")
    out = df.copy()
    present = [c for c in factor_columns if c in out.columns]
    group_cols = [c for c in group_columns if c in out.columns]
    if not present or not group_cols:
        for c in factor_columns:
            out[f"{c}{suffix}"] = np.nan
        return out

    for col in factor_columns:
        target = f"{col}{suffix}"
        if col not in out.columns:
            out[target] = np.nan
            continue
        values = pd.to_numeric(out[col], errors="coerce")

        def _standardize(s: pd.Series) -> pd.Series:
            valid = s.dropna()
            if valid.nunique() < 2 or len(valid) < min_symbols:
                return pd.Series(np.nan, index=s.index)
            if method == "zscore":
                mu = valid.mean()
                sigma = valid.std(ddof=0)
                if sigma == 0:
                    return pd.Series(np.nan, index=s.index)
                return (s - mu) / sigma
            # rank -> percentile in [-0.5, 0.5]
            return s.rank(pct=True) - 0.5

        out[target] = values.groupby([out[c] for c in group_cols]).transform(_standardize)
    return out


def standardized_columns(
    factor_columns: list[str] | tuple[str, ...], *, suffix: str = DEFAULT_SUFFIX
) -> list[str]:
    return [f"{c}{suffix}" for c in factor_columns]
