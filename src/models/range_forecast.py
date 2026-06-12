from __future__ import annotations

import pandas as pd

from src.features.labels import add_future_range_labels


def range_forecast_columns(horizons: list[int] | tuple[int, ...] = (1, 5, 10)) -> list[str]:
    columns: list[str] = []
    for horizon in horizons:
        columns.extend(
            [
                f"pred_next_{horizon}d_low_pct",
                f"pred_next_{horizon}d_high_pct",
                f"pred_next_{horizon}d_range_confidence",
            ]
        )
    return columns


def add_range_forecasts(
    frame: pd.DataFrame,
    *,
    horizons: list[int] | tuple[int, ...] = (1, 5, 10),
    window: int = 252,
    min_periods: int = 60,
    lower_quantile: float = 0.2,
    upper_quantile: float = 0.8,
    volatility_conditioning: bool = False,
    volatility_column: str = "atr_pct",
    volatility_window: int = 252,
) -> pd.DataFrame:
    """Add trailing empirical range forecasts without using unavailable labels.

    A label for row t and horizon h only becomes available at t+h. Shifting
    labels by h before rolling keeps current-day forecasts strictly causal.

    When volatility_conditioning is enabled, the unconditional rolling-quantile
    band is scaled by the ratio of current volatility (volatility_column,
    default atr_pct) to its own rolling mean. This turns the climatology
    baseline into a regime-aware band: calm regimes get narrower bands, high
    volatility regimes get wider ones. The scaling factor only uses data up to
    and including row t, so causality is preserved.
    """
    if frame.empty:
        output = frame.copy()
        for column in range_forecast_columns(horizons):
            output[column] = pd.Series(dtype=float)
        return output
    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError("Forecast quantiles must satisfy 0 <= lower < upper <= 1.")

    output = frame.sort_values(["symbol", "datetime"]).copy()
    labeled = add_future_range_labels(output, horizons=horizons)

    for horizon in horizons:
        low_label = f"next_{horizon}d_low_pct"
        high_label = f"next_{horizon}d_high_pct"
        available_low = labeled.groupby("symbol")[low_label].shift(horizon)
        available_high = labeled.groupby("symbol")[high_label].shift(horizon)

        predicted_low = available_low.groupby(output["symbol"]).transform(
            lambda values: values.rolling(window, min_periods=min_periods).quantile(
                lower_quantile
            )
        )
        predicted_high = available_high.groupby(output["symbol"]).transform(
            lambda values: values.rolling(window, min_periods=min_periods).quantile(
                upper_quantile
            )
        )
        observations = available_high.groupby(output["symbol"]).transform(
            lambda values: values.rolling(window, min_periods=1).count()
        )

        if volatility_conditioning and volatility_column in output.columns:
            vol = output[volatility_column]
            vol_mean = vol.groupby(output["symbol"]).transform(
                lambda values: values.rolling(
                    volatility_window, min_periods=min_periods
                ).mean()
            )
            scale = (vol / vol_mean).clip(lower=0.5, upper=2.5)
            predicted_low = predicted_low * scale
            predicted_high = predicted_high * scale

        output[f"pred_next_{horizon}d_low_pct"] = predicted_low
        output[f"pred_next_{horizon}d_high_pct"] = predicted_high
        output[f"pred_next_{horizon}d_range_confidence"] = (
            observations / float(window)
        ).clip(0, 1)

    return output
