from __future__ import annotations

"""Walk-forward learned forecast.

This replaces the hand-weighted score average with a model that *learns* how to
combine factors from data, while staying strictly causal:

- Target: the realized forward return over `horizon` trading rows per symbol.
- Walk-forward: the model is retrained periodically; when predicting date t it
  is trained ONLY on samples whose label was already realized before t
  (sample date-index j with j + horizon <= t). No future information leaks in.
- Default model is a numpy ridge regression (no extra dependency, deterministic,
  interpretable). If `model="auto"` and LightGBM is importable it is used
  instead; otherwise it falls back to ridge.

The output column `ml_pred_return_{h}d` is a predicted forward return (centred
at ~0, NOT a 0-100 score). It is measured -- not trusted -- by the validation
layer (signal_validation) just like every other signal.
"""

import numpy as np
import pandas as pd


def ml_forecast_column(horizon: int = 5) -> str:
    return f"ml_pred_return_{horizon}d"


def _fit_ridge(x_train: np.ndarray, y_train: np.ndarray, lam: float) -> np.ndarray:
    n, k = x_train.shape
    design = np.hstack([np.ones((n, 1)), x_train])
    reg = lam * np.eye(k + 1)
    reg[0, 0] = 0.0  # never penalise the intercept
    return np.linalg.solve(design.T @ design + reg, design.T @ y_train)


def _predict_ridge(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.hstack([np.ones((len(x), 1)), x]) @ weights


def _maybe_lightgbm():
    try:
        import lightgbm as lgb  # noqa: F401
        return lgb
    except Exception:
        return None


def add_ml_forecast(
    df: pd.DataFrame,
    feature_columns: list[str] | tuple[str, ...],
    *,
    horizon: int = 5,
    price_col: str = "close",
    retrain_every: int = 21,
    min_train_dates: int = 120,
    train_window_dates: int | None = None,
    ridge_lambda: float = 10.0,
    model: str = "ridge",
    min_train_rows: int = 50,
) -> pd.DataFrame:
    """Add a walk-forward predicted forward return. Causal by construction.

    Degrades safely: missing features/target, too little history, or fewer than
    `min_train_rows` realized samples leave predictions as NaN rather than
    raising or fabricating values.
    """
    out_col = ml_forecast_column(horizon)
    result = df.copy()
    result[out_col] = np.nan

    feats = [c for c in feature_columns if c in df.columns]
    needed = {"symbol", "datetime", price_col}
    if not feats or not needed.issubset(df.columns) or df.empty:
        return result

    work = df.copy().sort_values(["symbol", "datetime"])
    work["_target"] = work.groupby("symbol")[price_col].transform(
        lambda s: pd.to_numeric(s, errors="coerce").shift(-horizon)
        / pd.to_numeric(s, errors="coerce")
        - 1.0
    )

    date_list = sorted(work["datetime"].dropna().unique())
    if len(date_list) <= min_train_dates + horizon:
        return result  # not enough history to train and predict out-of-sample
    date_to_i = {d: i for i, d in enumerate(date_list)}
    work["_di"] = work["datetime"].map(date_to_i)

    x_raw = work[feats].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x_raw)
    # A missing standardized factor means "no peer signal today" -> impute its
    # neutral value 0 (the cross-sectional mean). This keeps rows usable when
    # some factors are structurally absent (e.g. chip/options on a mock run or
    # before they are collected) instead of dropping every such row.
    x_all = np.where(finite, x_raw, 0.0)
    has_any_factor = finite.any(axis=1)
    y_all = work["_target"].to_numpy(dtype=float)
    di_all = work["_di"].to_numpy()
    preds = np.full(len(work), np.nan)

    lgb = _maybe_lightgbm() if model in ("auto", "lightgbm") else None

    n_dates = len(date_list)
    retrain_points = list(range(min_train_dates, n_dates, max(1, retrain_every)))
    for idx, r in enumerate(retrain_points):
        next_r = retrain_points[idx + 1] if idx + 1 < len(retrain_points) else n_dates
        # Training rows: label realized strictly before r (j + horizon <= r).
        train_mask = di_all <= (r - horizon)
        if train_window_dates is not None:
            train_mask &= di_all >= (r - horizon - train_window_dates)
        rows_ok = train_mask & np.isfinite(y_all) & has_any_factor
        if rows_ok.sum() < min_train_rows:
            continue
        x_tr, y_tr = x_all[rows_ok], y_all[rows_ok]

        if lgb is not None:
            booster = lgb.LGBMRegressor(
                n_estimators=200, num_leaves=15, learning_rate=0.05,
                min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                verbose=-1,
            )
            booster.fit(x_tr, y_tr)
            predict_fn = booster.predict
        else:
            weights = _fit_ridge(x_tr, y_tr, ridge_lambda)
            predict_fn = lambda x, w=weights: _predict_ridge(w, x)

        # Predict the out-of-sample block [r, next_r).
        block = (di_all >= r) & (di_all < next_r) & has_any_factor
        if block.any():
            preds[block] = predict_fn(x_all[block])

    pred_series = pd.Series(preds, index=work.index)
    result[out_col] = pred_series.reindex(result.index)
    return result
