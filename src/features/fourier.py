from __future__ import annotations

import numpy as np
import pandas as pd


def _dominant_cycle(values: np.ndarray, min_period: int = 5, max_period: int = 80) -> tuple[float, float]:
    """Return dominant period and relative spectral strength.

    The input is a 1D array of returns. This is a simplified Fourier feature,
    not a trading signal by itself.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < max(20, min_period * 2):
        return np.nan, np.nan
    x = values - np.nanmean(values)
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    periods = np.divide(1.0, freqs, out=np.full_like(freqs, np.inf), where=freqs > 0)
    mask = (periods >= min_period) & (periods <= max_period)
    if not np.any(mask):
        return np.nan, np.nan
    masked_power = spectrum[mask]
    if masked_power.sum() <= 0:
        return np.nan, np.nan
    idx = np.argmax(masked_power)
    dominant_period = periods[mask][idx]
    strength = masked_power[idx] / spectrum[1:].sum() if spectrum[1:].sum() > 0 else np.nan
    return float(dominant_period), float(strength)


def add_fourier_features(df: pd.DataFrame, window: int = 120, min_period: int = 5, max_period: int = 80) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.sort_values(["symbol", "datetime"]).copy()
    out["fourier_main_cycle"] = np.nan
    out["fourier_cycle_strength"] = np.nan

    for symbol, idx in out.groupby("symbol").groups.items():
        symbol_df = out.loc[idx].sort_values("datetime")
        returns = symbol_df["return_1d"].to_numpy()
        cycles = []
        strengths = []
        for i in range(len(symbol_df)):
            start = max(0, i - window + 1)
            period, strength = _dominant_cycle(returns[start : i + 1], min_period=min_period, max_period=max_period)
            cycles.append(period)
            strengths.append(strength)
        out.loc[symbol_df.index, "fourier_main_cycle"] = cycles
        out.loc[symbol_df.index, "fourier_cycle_strength"] = strengths
    return out
