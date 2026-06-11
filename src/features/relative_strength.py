from __future__ import annotations

import numpy as np
import pandas as pd


def add_relative_strength(df: pd.DataFrame, benchmark_map: dict[str, str | None]) -> pd.DataFrame:
    """Add relative strength against configured benchmark using 20d return.

    If benchmark data is unavailable, relative strength stays NaN.
    """
    if df.empty:
        return df.copy()
    out = df.copy()
    out["relative_strength_20d"] = np.nan

    latest_by_symbol = out.pivot_table(index="datetime", columns="symbol", values="return_20d", aggfunc="last")
    for symbol, benchmark in benchmark_map.items():
        if not benchmark or symbol not in latest_by_symbol.columns or benchmark not in latest_by_symbol.columns:
            continue
        rs = latest_by_symbol[symbol] - latest_by_symbol[benchmark]
        rs_frame = rs.rename("relative_strength_20d").reset_index()
        mask = out["symbol"].eq(symbol)
        out_symbol = out.loc[mask, ["datetime"]].merge(rs_frame, on="datetime", how="left")
        out.loc[mask, "relative_strength_20d"] = out_symbol["relative_strength_20d"].to_numpy()
    return out
