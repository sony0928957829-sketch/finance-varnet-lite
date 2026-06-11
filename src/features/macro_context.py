from __future__ import annotations

import numpy as np
import pandas as pd


MACRO_SYMBOLS = {"^VIX", "^TNX", "DX-Y.NYB", "TWD=X"}


def add_macro_context(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        output = frame.copy()
        output["macro_risk_score"] = pd.Series(dtype=float)
        output["cross_market_divergence_score"] = pd.Series(dtype=float)
        return output

    output = frame.sort_values(["symbol", "datetime"]).copy()
    dates = pd.Index(sorted(output["datetime"].dropna().unique()), name="datetime")
    latest = output.pivot_table(
        index="datetime",
        columns="symbol",
        values=["close", "return_20d"],
        aggfunc="last",
    ).reindex(dates).ffill(limit=5)

    components: list[pd.Series] = []
    if ("close", "^VIX") in latest.columns:
        components.append(((latest[("close", "^VIX")] - 12.0) * 4.0).clip(0, 100))
    if ("return_20d", "^TNX") in latest.columns:
        components.append((50 + latest[("return_20d", "^TNX")] * 250).clip(0, 100))
    if ("return_20d", "DX-Y.NYB") in latest.columns:
        components.append((50 + latest[("return_20d", "DX-Y.NYB")] * 500).clip(0, 100))
    if ("return_20d", "TWD=X") in latest.columns:
        components.append((50 + latest[("return_20d", "TWD=X")] * 500).clip(0, 100))

    if components:
        macro_risk = pd.concat(components, axis=1).mean(axis=1, skipna=True)
        output = output.merge(
            macro_risk.rename("macro_risk_score").reset_index(),
            on="datetime",
            how="left",
        )
    else:
        output["macro_risk_score"] = np.nan

    output["cross_market_divergence_score"] = np.nan
    if ("return_20d", "TAIEX") in latest.columns:
        taiex = latest[("return_20d", "TAIEX")]
        for symbol in output.loc[output["market"].eq("TW"), "symbol"].unique():
            if symbol == "TAIEX" or ("return_20d", symbol) not in latest.columns:
                continue
            divergence = (
                latest[("return_20d", symbol)].sub(taiex).abs().mul(500).clip(0, 100)
            )
            mapped = output.loc[output["symbol"].eq(symbol), "datetime"].map(divergence)
            output.loc[output["symbol"].eq(symbol), "cross_market_divergence_score"] = (
                mapped.to_numpy()
            )
    return output
