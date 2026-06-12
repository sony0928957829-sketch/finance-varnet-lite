from __future__ import annotations

import numpy as np
import pandas as pd


def _dominant_cycle(values: np.ndarray, min_period: int = 5, max_period: int = 80) -> tuple[float, float, bool]:
    """Return dominant period and relative spectral strength.

    The input is a 1D array of returns. This is a simplified Fourier feature,
    not a trading signal by itself.
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < max(20, min_period * 2):
        return np.nan, np.nan, False
    x = values - np.nanmean(values)
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)
    periods = np.divide(1.0, freqs, out=np.full_like(freqs, np.inf), where=freqs > 0)
    mask = (periods >= min_period) & (periods <= max_period)
    if not np.any(mask):
        return np.nan, np.nan, False
    masked_power = spectrum[mask]
    if masked_power.sum() <= 0:
        return np.nan, np.nan, False
    idx = np.argmax(masked_power)
    dominant_period = periods[mask][idx]
    strength = masked_power[idx] / spectrum[1:].sum() if spectrum[1:].sum() > 0 else np.nan
    # With window n, periods near n have very coarse FFT resolution: adjacent
    # bins around period p are ~p^2/n apart. Flag the estimate unreliable when
    # the dominant period exceeds n/3 (fewer than ~3 full cycles observed).
    reliable = bool(dominant_period <= n / 3)
    return float(dominant_period), float(strength), reliable


def add_fourier_features(df: pd.DataFrame, window: int = 120, min_period: int = 5, max_period: int = 80, stride: int = 1) -> pd.DataFrame:
    """stride > 1 computes the spectrum every `stride` rows and forward-fills
    between them; an O(stride) speedup with slightly stale values in between.
    Use stride=1 (default) for research, larger values for wide watchlists."""
    if df.empty:
        return df.copy()
    out = df.sort_values(["symbol", "datetime"]).copy()
    out["fourier_main_cycle"] = np.nan
    out["fourier_cycle_strength"] = np.nan
    out["fourier_cycle_reliable"] = False

    for symbol, idx in out.groupby("symbol").groups.items():
        symbol_df = out.loc[idx].sort_values("datetime")
        returns = symbol_df["return_1d"].to_numpy()
        cycles = []
        strengths = []
        reliable_flags = []
        last = (np.nan, np.nan, False)
        for i in range(len(symbol_df)):
            if i % stride == 0 or i == len(symbol_df) - 1:
                start = max(0, i - window + 1)
                last = _dominant_cycle(returns[start : i + 1], min_period=min_period, max_period=max_period)
            period, strength, reliable = last
            cycles.append(period)
            strengths.append(strength)
            reliable_flags.append(reliable)
        out.loc[symbol_df.index, "fourier_main_cycle"] = cycles
        out.loc[symbol_df.index, "fourier_cycle_strength"] = strengths
        out.loc[symbol_df.index, "fourier_cycle_reliable"] = reliable_flags
    return out
