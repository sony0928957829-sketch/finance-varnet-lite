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
) -> pd.DataFrame:
    """Add trailing empirical range forecasts without using unavailable labels.

    A label for row t and horizon h only becomes available at t+h. Shifting
    labels by h before rolling keeps current-day forecasts strictly causal.
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

        output[f"pred_next_{horizon}d_low_pct"] = predicted_low
        output[f"pred_next_{horizon}d_high_pct"] = predicted_high
        output[f"pred_next_{horizon}d_range_confidence"] = (
            observations / float(window)
        ).clip(0, 1)

    return output
