from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_zscore(value: float, history: np.ndarray) -> float:
    history = history[np.isfinite(history)]
    if len(history) < 20:
        return np.nan
    mu = history.mean()
    sigma = history.std(ddof=0)
    if sigma == 0:
        return 0.0
    return float((value - mu) / sigma)


def _wavelet_energy(values: np.ndarray, wavelet_name: str = "db4", level: int = 3) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 32:
        return np.nan, np.nan, np.nan
    x = values - np.nanmean(values)
    try:
        import pywt
        coeffs = pywt.wavedec(x, wavelet=wavelet_name, level=min(level, pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet_name).dec_len)))
        details = coeffs[1:]
        energies = [float(np.sum(c ** 2)) for c in details]
        total = sum(energies) if sum(energies) > 0 else np.nan
        if not np.isfinite(total):
            return np.nan, np.nan, np.nan
        # Approximate short/mid/long detail energy ratios.
        padded = (energies + [np.nan, np.nan, np.nan])[:3]
        return padded[0] / total, padded[1] / total, padded[2] / total
    except Exception:
        # Fallback if PyWavelets is unavailable: use multi-window volatility ratios.
        short = np.std(x[-5:]) if len(x) >= 5 else np.nan
        mid = np.std(x[-20:]) if len(x) >= 20 else np.nan
        long = np.std(x) if len(x) >= 32 else np.nan
        total = np.nansum([short, mid, long])
        if total == 0 or not np.isfinite(total):
            return np.nan, np.nan, np.nan
        return short / total, mid / total, long / total


def add_wavelet_features(df: pd.DataFrame, window: int = 120, wavelet_name: str = "db4", level: int = 3, stride: int = 1) -> pd.DataFrame:
    """stride > 1 recomputes wavelet energy every `stride` rows and reuses the
    last value in between; an O(stride) speedup for wide watchlists."""
    if df.empty:
        return df.copy()
    out = df.sort_values(["symbol", "datetime"]).copy()
    out["wavelet_energy_short"] = np.nan
    out["wavelet_energy_mid"] = np.nan
    out["wavelet_energy_long"] = np.nan
    out["wavelet_anomaly_score"] = np.nan

    for symbol, idx in out.groupby("symbol").groups.items():
        symbol_df = out.loc[idx].sort_values("datetime")
        returns = symbol_df["return_1d"].to_numpy()
        short_e, mid_e, long_e, anomaly = [], [], [], []
        baseline_energy = []
        # Energies computed over short warmup windows are unstable; keeping
        # them in the z-score baseline contaminates early anomaly scores.
        min_window_for_baseline = min(window, 64)
        last = (np.nan, np.nan, np.nan)
        for i in range(len(symbol_df)):
            start = max(0, i - window + 1)
            effective_window = i + 1 - start
            if i % stride == 0 or i == len(symbol_df) - 1:
                last = _wavelet_energy(returns[start : i + 1], wavelet_name=wavelet_name, level=level)
            es, em, el = last
            short_e.append(es)
            mid_e.append(em)
            long_e.append(el)
            current_energy = np.nansum([es, em])
            z = _rolling_zscore(current_energy, np.asarray(baseline_energy, dtype=float))
            anomaly.append(np.clip(abs(z) * 20, 0, 100) if np.isfinite(z) else np.nan)
            if effective_window >= min_window_for_baseline:
                baseline_energy.append(current_energy)

        out.loc[symbol_df.index, "wavelet_energy_short"] = short_e
        out.loc[symbol_df.index, "wavelet_energy_mid"] = mid_e
        out.loc[symbol_df.index, "wavelet_energy_long"] = long_e
        out.loc[symbol_df.index, "wavelet_anomaly_score"] = anomaly
    return out
