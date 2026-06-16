from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.factor_standardize import (
    add_standardized_factors,
    standardized_columns,
)


def _panel():
    dates = pd.date_range("2025-01-01", periods=3, freq="D")
    rows = []
    for d in dates:
        for mk, base in [("TW", 0.0), ("US", 100.0)]:
            for i, sym in enumerate(["A", "B", "C", "D"]):
                rows.append({"datetime": d, "market": mk, "symbol": sym + mk, "f": base + i})
    return pd.DataFrame(rows)


def test_zscore_is_mean_zero_per_group():
    out = add_standardized_factors(_panel(), ["f"], group_columns=("datetime", "market"))
    means = out.groupby(["datetime", "market"])["f_cs"].mean().abs()
    assert (means < 1e-9).all()


def test_missing_column_becomes_nan_not_error():
    out = add_standardized_factors(_panel(), ["does_not_exist"])
    assert out["does_not_exist_cs"].isna().all()


def test_thin_cross_section_degrades_to_nan():
    # Only one symbol on the date -> cannot standardize against peers.
    df = pd.DataFrame({"datetime": ["2025-01-01"], "market": ["TW"], "symbol": ["A"], "f": [1.0]})
    out = add_standardized_factors(df, ["f"], min_symbols=3)
    assert out["f_cs"].isna().all()


def test_rank_method_in_expected_range():
    out = add_standardized_factors(_panel(), ["f"], method="rank")
    vals = out["f_cs"].dropna()
    assert vals.between(-0.5, 0.5).all()


def test_standardized_columns_names():
    assert standardized_columns(["a", "b"]) == ["a_cs", "b_cs"]
