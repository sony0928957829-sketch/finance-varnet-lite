from __future__ import annotations

import numpy as np
import pandas as pd


MACRO_SYMBOLS = {"^VIX", "^TNX", "DX-Y.NYB", "TWD=X"}

# Markets whose daily close happens BEFORE the same calendar date's US close.
# For these markets, same-date US-session macro values are future information
# at their own close, so US macro series must be lagged by one row.
# TW closes 13:30 Asia/Taipei; the same-date US close lands ~04:00 next-day Taipei time.
US_SESSION_MACRO_SYMBOLS = {"^VIX", "^TNX", "DX-Y.NYB", "TWD=X"}
MARKETS_CLOSING_BEFORE_US = {"TW"}


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

    def _macro_components(table: pd.DataFrame) -> list[pd.Series]:
        parts: list[pd.Series] = []
        if ("close", "^VIX") in table.columns:
            parts.append(((table[("close", "^VIX")] - 12.0) * 4.0).clip(0, 100))
        if ("return_20d", "^TNX") in table.columns:
            parts.append((50 + table[("return_20d", "^TNX")] * 250).clip(0, 100))
        if ("return_20d", "DX-Y.NYB") in table.columns:
            parts.append((50 + table[("return_20d", "DX-Y.NYB")] * 500).clip(0, 100))
        if ("return_20d", "TWD=X") in table.columns:
            parts.append((50 + table[("return_20d", "TWD=X")] * 500).clip(0, 100))
        return parts

    # Lagged view: US-session macro shifted one row so early-closing markets
    # (e.g. TW) never see a US close that has not happened yet.
    lagged = latest.copy()
    for values_name in ("close", "return_20d"):
        for symbol in US_SESSION_MACRO_SYMBOLS:
            column = (values_name, symbol)
            if column in lagged.columns:
                lagged[column] = lagged[column].shift(1)

    components = _macro_components(latest)
    lagged_components = _macro_components(lagged)

    if components:
        macro_risk = pd.concat(components, axis=1).mean(axis=1, skipna=True)
        macro_risk_lagged = pd.concat(lagged_components, axis=1).mean(axis=1, skipna=True)
        risk_frame = pd.concat(
            [
                macro_risk.rename("macro_risk_score_current"),
                macro_risk_lagged.rename("macro_risk_score_lagged"),
            ],
            axis=1,
        ).reset_index()
        output = output.merge(risk_frame, on="datetime", how="left")
        use_lagged = output["market"].isin(MARKETS_CLOSING_BEFORE_US)
        output["macro_risk_score"] = np.where(
            use_lagged,
            output["macro_risk_score_lagged"],
            output["macro_risk_score_current"],
        )
        output = output.drop(
            columns=["macro_risk_score_current", "macro_risk_score_lagged"]
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
