from __future__ import annotations

"""Coverage backtest for range forecasts.

This is the yardstick every future range model (v0.2+) must beat. For each
symbol and horizon it measures:

- coverage: fraction of realized [low, high] ranges fully inside the
  predicted [pred_low, pred_high] band. With 0.2/0.8 quantile bands the
  joint coverage target is empirical; what matters is per-side calibration.
- low_side_coverage / high_side_coverage: P(realized_low >= pred_low) and
  P(realized_high <= pred_high). For 0.2/0.8 quantiles each side should be
  close to 0.80 if the forecaster is calibrated.
- mean_band_width: average predicted band width. Two calibrated models are
  ranked by the narrower band (sharper while staying calibrated).

Evaluation only uses rows where both prediction and realized label exist,
so the warmup period and the final h rows are excluded automatically.
"""

import pandas as pd


def evaluate_range_coverage(
    frame: pd.DataFrame,
    horizons: list[int] | tuple[int, ...] = (1, 5, 10),
) -> pd.DataFrame:
    """Return a tidy DataFrame: one row per (symbol, horizon) with metrics.

    Expects columns produced by add_range_forecasts (pred_next_{h}d_low_pct,
    pred_next_{h}d_high_pct) and add_future_range_labels
    (next_{h}d_low_pct, next_{h}d_high_pct).
    """
    records: list[dict] = []
    for symbol, group in frame.groupby("symbol"):
        for horizon in horizons:
            pred_low = group.get(f"pred_next_{horizon}d_low_pct")
            pred_high = group.get(f"pred_next_{horizon}d_high_pct")
            real_low = group.get(f"next_{horizon}d_low_pct")
            real_high = group.get(f"next_{horizon}d_high_pct")
            if pred_low is None or real_low is None:
                continue

            valid = (
                pred_low.notna()
                & pred_high.notna()
                & real_low.notna()
                & real_high.notna()
            )
            n = int(valid.sum())
            if n == 0:
                continue

            p_low = pred_low[valid]
            p_high = pred_high[valid]
            r_low = real_low[valid]
            r_high = real_high[valid]

            low_ok = r_low >= p_low
            high_ok = r_high <= p_high
            records.append(
                {
                    "symbol": symbol,
                    "horizon": horizon,
                    "n_observations": n,
                    "coverage": float((low_ok & high_ok).mean()),
                    "low_side_coverage": float(low_ok.mean()),
                    "high_side_coverage": float(high_ok.mean()),
                    "mean_band_width": float((p_high - p_low).mean()),
                    "mean_realized_width": float((r_high - r_low).mean()),
                }
            )
    return pd.DataFrame.from_records(records)
