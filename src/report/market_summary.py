from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def classify_market_state(overall_risk: float, overall_condition: float) -> str:
    if not np.isfinite(overall_risk) or not np.isfinite(overall_condition):
        return "資料不足"
    if overall_risk >= 60:
        return "高波動"
    if overall_condition >= 75:
        return "強勢"
    if overall_condition >= 60:
        return "偏多"
    if overall_condition >= 45:
        return "中性"
    if overall_condition >= 30:
        return "轉弱"
    return "偏弱"


def _ranked_symbols(
    latest: pd.DataFrame,
    column: str,
    *,
    ascending: bool,
    limit: int,
    exclude: set[str] | None = None,
) -> list[str]:
    exclude = exclude or set()
    ranked = latest.dropna(subset=[column]).sort_values(
        [column, "symbol"],
        ascending=[ascending, True],
    )
    return [
        str(symbol)
        for symbol in ranked["symbol"]
        if str(symbol) not in exclude
    ][:limit]


def build_market_summary(
    latest: pd.DataFrame,
    health_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic market observation summary from latest features."""
    if latest.empty:
        return {
            "market_state": "資料不足",
            "overall_risk": np.nan,
            "overall_condition": np.nan,
            "abnormal_symbols": [],
            "strong_symbols": [],
            "weak_symbols": [],
            "watch_points": ["目前沒有足夠資料可產生市場摘要。"],
            "data_health": "資料不足",
        }

    overall_risk = float(latest["risk_score"].mean(skipna=True))
    overall_condition = float(latest["condition_score"].mean(skipna=True))
    abnormal_symbols = _ranked_symbols(
        latest,
        "risk_score",
        ascending=False,
        limit=min(3, len(latest)),
    )
    ranking_limit = min(3, max(1, len(latest) // 2))
    strong_symbols = _ranked_symbols(
        latest,
        "condition_score",
        ascending=False,
        limit=ranking_limit,
    )
    weak_symbols = _ranked_symbols(
        latest,
        "condition_score",
        ascending=True,
        limit=ranking_limit,
        exclude=set(strong_symbols),
    )

    watch_points: list[str] = []
    if abnormal_symbols:
        watch_points.append(
            f"優先追蹤 {abnormal_symbols[0]} 的波動與異常風險是否持續升高。"
        )
    if weak_symbols:
        watch_points.append(
            f"觀察 {weak_symbols[0]} 能否停止轉弱，並修復短中期動能。"
        )

    market_returns = (
        latest.dropna(subset=["return_1d"])
        .groupby("market")["return_1d"]
        .mean()
        .sort_values()
    )
    if len(market_returns) >= 2:
        weakest_market = str(market_returns.index[0])
        strongest_market = str(market_returns.index[-1])
        spread = float(market_returns.iloc[-1] - market_returns.iloc[0])
        if spread >= 0.02:
            watch_points.append(
                f"{strongest_market} 與 {weakest_market} 單日表現差距約 "
                f"{spread * 100:.2f}%，留意跨市場背離。"
            )
        else:
            watch_points.append("跨市場單日表現差距有限，持續觀察是否同步轉強或轉弱。")
    else:
        watch_points.append("目前跨市場資料不足，暫不判定市場背離。")

    if health_report is None:
        data_health = "未提供"
    else:
        health_status = health_report.get("status", "unknown")
        warning_count = health_report.get("summary", {}).get("warning_count", 0)
        data_health = {
            "healthy": "正常",
            "warning": f"警告（{warning_count} 項）",
            "error": "錯誤",
        }.get(health_status, "未知")
        if health_status == "warning":
            watch_points.append("資料健康檢查有警告，解讀訊號時需保留不確定性。")

    return {
        "market_state": classify_market_state(overall_risk, overall_condition),
        "overall_risk": overall_risk,
        "overall_condition": overall_condition,
        "abnormal_symbols": abnormal_symbols,
        "strong_symbols": strong_symbols,
        "weak_symbols": weak_symbols,
        "watch_points": watch_points,
        "data_health": data_health,
    }
