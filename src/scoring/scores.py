from __future__ import annotations

import numpy as np
import pandas as pd


def _clip_score(value: float) -> float:
    if not np.isfinite(value):
        return np.nan
    return float(np.clip(value, 0, 100))


def _trend_score(row: pd.Series) -> float:
    # Return NaN (-> "資料不足") instead of a fake-low score during warmup.
    required = ["close", "ma_20", "ma_60", "return_20d", "return_60d"]
    values = [row.get(name, np.nan) for name in required]
    if not all(np.isfinite(v) for v in values):
        return np.nan
    close, ma_20, ma_60, return_20d, return_60d = values
    score = 0
    if close > ma_20:
        score += 25
    if close > ma_60:
        score += 25
    if ma_20 > ma_60:
        score += 20
    if return_20d > 0:
        score += 15
    if return_60d > 0:
        score += 15
    return _clip_score(score)


def _momentum_score(row: pd.Series) -> float:
    # Use non-overlapping windows so one large day is not triple-counted:
    # r1 = day 1, r_2_5 = days 2-5, r_6_20 = days 6-20.
    r1 = row.get("return_1d", np.nan)
    r5 = row.get("return_5d", np.nan)
    r20 = row.get("return_20d", np.nan)
    if not all(np.isfinite(v) for v in (r1, r5, r20)):
        return np.nan
    r_2_5 = (1 + r5) / (1 + r1) - 1
    r_6_20 = (1 + r20) / (1 + r5) - 1
    raw = 50 + 500 * r1 + 300 * r_2_5 + 150 * r_6_20
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


# Index/macro daily volume from free sources is frequently zero or unreliable;
# a volume score computed from it is noise, so it is suppressed for these symbols.
SYMBOLS_WITHOUT_RELIABLE_VOLUME = {"TAIEX", "^TWII", "^VIX", "^TNX", "DX-Y.NYB", "TWD=X"}


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["trend_score"] = out.apply(_trend_score, axis=1)
    out["momentum_score"] = out.apply(_momentum_score, axis=1)
    out["volume_score"] = out.apply(_volume_score, axis=1)
    out.loc[out["symbol"].isin(SYMBOLS_WITHOUT_RELIABLE_VOLUME), "volume_score"] = np.nan
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
    # Chip/options sentiment signals are directional too (0-100, 50 neutral),
    # so they join the average when present; absent -> NaN -> skipped.
    directional_columns = [
        "trend_score",
        "momentum_score",
        "relative_strength_score",
        "institutional_flow_score",
        "options_sentiment_score",
    ]
    available_directional = [c for c in directional_columns if c in out.columns]
    out["condition_score"] = out[available_directional].mean(axis=1, skipna=True)
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
