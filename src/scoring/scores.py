from __future__ import annotations

import numpy as np
import pandas as pd


def _clip_score(value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    return float(np.clip(value, 0, 100))


def _trend_score(row: pd.Series) -> float:
    score = 0
    close = row.get("close", np.nan)
    if close > row.get("ma_20", np.inf):
        score += 25
    if close > row.get("ma_60", np.inf):
        score += 25
    if row.get("ma_20", -np.inf) > row.get("ma_60", np.inf):
        score += 20
    if row.get("return_20d", -np.inf) > 0:
        score += 15
    if row.get("return_60d", -np.inf) > 0:
        score += 15
    return _clip_score(score)


def _momentum_score(row: pd.Series) -> float:
    r1 = row.get("return_1d", 0.0)
    r5 = row.get("return_5d", 0.0)
    r20 = row.get("return_20d", 0.0)
    raw = 50 + 500 * r1 + 250 * r5 + 100 * r20
    return _clip_score(raw)


def _volume_score(row: pd.Series) -> float:
    ratio = row.get("volume_ratio", np.nan)
    if not np.isfinite(ratio):
        return np.nan
    if 0.8 <= ratio <= 1.5:
        return 50
    if ratio > 1.5:
        return _clip_score(50 + (ratio - 1.5) * 25)
    return _clip_score(50 - (0.8 - ratio) * 30)


def _volatility_risk_score(row: pd.Series) -> float:
    # Higher means more risk, not more bullishness.
    atr_pct = row.get("atr_pct", np.nan)
    range_pct = row.get("range_pct", np.nan)
    vol = row.get("volatility_20", np.nan)
    raw = 0
    if np.isfinite(atr_pct):
        raw += atr_pct * 1200
    if np.isfinite(range_pct):
        raw += range_pct * 700
    if np.isfinite(vol):
        raw += vol * 800
    return _clip_score(raw)


def _relative_strength_score(row: pd.Series) -> float:
    rs = row.get("relative_strength_20d", np.nan)
    if not np.isfinite(rs):
        return np.nan
    return _clip_score(50 + rs * 500)


def _anomaly_risk_score(row: pd.Series) -> float:
    wavelet = row.get("wavelet_anomaly_score", np.nan)
    cycle_strength = row.get("fourier_cycle_strength", np.nan)
    vol_score = _volatility_risk_score(row)
    parts = []
    if np.isfinite(wavelet):
        parts.append(wavelet)
    if np.isfinite(cycle_strength):
        # Strong cycle is not automatically risky, but a very concentrated cycle can mean crowded rhythm.
        parts.append(min(cycle_strength * 150, 100))
    if np.isfinite(vol_score):
        parts.append(vol_score)
    if not parts:
        return np.nan
    return _clip_score(np.nanmean(parts))


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["trend_score"] = out.apply(_trend_score, axis=1)
    out["momentum_score"] = out.apply(_momentum_score, axis=1)
    out["volume_score"] = out.apply(_volume_score, axis=1)
    out["volatility_risk_score"] = out.apply(_volatility_risk_score, axis=1)
    out["relative_strength_score"] = out.apply(_relative_strength_score, axis=1)
    out["anomaly_risk_score"] = out.apply(_anomaly_risk_score, axis=1)

    # Overall risk focuses on downside/instability signals, not bullishness.
    risk_columns = [
        "volatility_risk_score",
        "anomaly_risk_score",
        "macro_risk_score",
        "cross_market_divergence_score",
    ]
    available_risk_columns = [column for column in risk_columns if column in out.columns]
    out["risk_score"] = (
        out[available_risk_columns]
        .mean(axis=1, skipna=True)
        .clip(0, 100)
    )

    # Market condition score is directional/quality, not a buy signal.
    out["condition_score"] = out[["trend_score", "momentum_score", "relative_strength_score"]].mean(axis=1, skipna=True)
    out["risk_level"] = out["risk_score"].apply(classify_risk)
    out["condition_label"] = out["condition_score"].apply(classify_condition)
    return out


def classify_risk(score: float) -> str:
    if not np.isfinite(score):
        return "資料不足"
    if score < 35:
        return "低"
    if score < 60:
        return "中"
    if score < 80:
        return "高"
    return "極高"


def classify_condition(score: float) -> str:
    if not np.isfinite(score):
        return "資料不足"
    if score >= 75:
        return "強勢"
    if score >= 60:
        return "偏多"
    if score >= 45:
        return "中性"
    if score >= 30:
        return "轉弱"
    return "偏弱"
