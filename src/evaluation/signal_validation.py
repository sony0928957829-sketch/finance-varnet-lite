from __future__ import annotations

"""Self-validation for directional signals.

This module answers one question, honestly: do the 0-100 scores (sentiment,
flow, condition, ...) actually have predictive edge over future returns, or
are they noise? It does NOT produce buy/sell advice. It produces a scorecard
the user reads to decide how much, if at all, to trust a signal.

Two metrics per (signal, horizon):

- direction_hit_rate: among rows where the signal takes a side
  (score != neutral), the fraction where sign(score - neutral) matches the
  sign of the realized forward return. 0.50 == coin flip. A binomial z-stat
  vs 0.50 says whether it beats random.
- ic (information coefficient): pooled Spearman rank correlation between the
  signal and the realized forward return. ~0 == no monotonic edge. A t-stat
  says whether it is distinguishable from zero.

Both are computed only on rows where BOTH the signal and the realized forward
return exist, so warmup rows and the final `horizon` rows (no future yet) are
excluded automatically. With too few usable rows, metrics degrade to NaN and
the user is told the sample is too small rather than shown a fake number.

`update_track_record` appends one dated batch of scorecard rows to a parquet
history, so running this every day accumulates evidence and the edge (or lack
of it) can be tracked through time instead of judged off a single snapshot.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Below this many usable observations, a hit rate / IC is not worth reporting.
_MIN_OBS = 20


def add_forward_returns(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (5,),
    price_col: str = "close",
) -> pd.DataFrame:
    """Add realized forward returns per symbol: fwd_return_{h}d.

    fwd_return_{h}d at row t = close[t+h] / close[t] - 1, computed within each
    symbol after sorting by datetime. The last h rows of each symbol get NaN
    (no future to measure against yet).
    """
    out = frame.copy()
    needed = {"symbol", "datetime", price_col}
    if not needed.issubset(out.columns):
        for h in horizons:
            out[f"fwd_return_{h}d"] = np.nan
        return out

    out = out.sort_values(["symbol", "datetime"])
    price = pd.to_numeric(out[price_col], errors="coerce")
    out = out.assign(**{price_col: price})
    for h in horizons:
        future = out.groupby("symbol")[price_col].shift(-h)
        out[f"fwd_return_{h}d"] = future / out[price_col] - 1.0
    return out


def _spearman_ic(signal: pd.Series, fwd_return: pd.Series) -> float:
    """Pooled Spearman rank correlation. NaN if not enough variation/data.

    Computed as Pearson correlation of the ranks (rank-then-Pearson) so it
    needs only numpy/pandas, no scipy dependency.
    """
    if signal.nunique(dropna=True) < 2 or fwd_return.nunique(dropna=True) < 2:
        return np.nan
    return float(signal.rank().corr(fwd_return.rank()))  # Pearson on ranks == Spearman


def _ic_tstat(ic: float, n: int) -> float:
    """t-stat for a correlation being != 0. |t| >~ 2 ~ significant at ~5%."""
    if not np.isfinite(ic) or n < 3 or abs(ic) >= 1.0:
        return np.nan
    return float(ic * np.sqrt((n - 2) / (1.0 - ic**2)))


def _hit_zstat(hit_rate: float, n: int) -> float:
    """z-stat for hit rate != 0.50 under a fair-coin null. |z| >~ 2 ~ sig."""
    if not np.isfinite(hit_rate) or n < 1:
        return np.nan
    return float((hit_rate - 0.5) / np.sqrt(0.25 / n))


def evaluate_signals(
    frame: pd.DataFrame,
    score_columns: list[str] | tuple[str, ...],
    *,
    horizons: tuple[int, ...] = (5,),
    price_col: str = "close",
    neutral: float | None = 50.0,
    min_obs: int = _MIN_OBS,
) -> pd.DataFrame:
    """Return a tidy scorecard: one row per (signal, horizon).

    Columns: signal, horizon, n, direction_hit_rate, direction_n, hit_zstat,
    ic, ic_tstat. Rows with too few usable observations report NaN metrics but
    still report n, so "not enough data yet" is visible rather than hidden.

    `neutral` is the score value that means "no side" (50 for the 0-100
    scores). Pass neutral=None to skip the direction test for signals that are
    not centered (e.g. a raw put_call_ratio).
    """
    work = add_forward_returns(frame, horizons=horizons, price_col=price_col)
    records: list[dict] = []

    for signal in score_columns:
        if signal not in work.columns:
            # Signal absent entirely -> report it as unavailable, don't crash.
            for h in horizons:
                records.append({
                    "signal": signal, "horizon": h, "n": 0,
                    "direction_hit_rate": np.nan, "direction_n": 0,
                    "hit_zstat": np.nan, "ic": np.nan, "ic_tstat": np.nan,
                })
            continue

        sig_all = pd.to_numeric(work[signal], errors="coerce")
        for h in horizons:
            fwd = work[f"fwd_return_{h}d"]
            valid = sig_all.notna() & fwd.notna()
            n = int(valid.sum())
            sig = sig_all[valid]
            ret = fwd[valid]

            if n < min_obs:
                records.append({
                    "signal": signal, "horizon": h, "n": n,
                    "direction_hit_rate": np.nan, "direction_n": 0,
                    "hit_zstat": np.nan, "ic": np.nan, "ic_tstat": np.nan,
                })
                continue

            ic = _spearman_ic(sig, ret)

            if neutral is None:
                hit_rate, dir_n, z = np.nan, 0, np.nan
            else:
                sided = sig != neutral
                dir_n = int(sided.sum())
                if dir_n == 0:
                    hit_rate, z = np.nan, np.nan
                else:
                    # Forward return exactly 0 counts as a miss (no move caught).
                    correct = np.sign(sig[sided] - neutral) == np.sign(ret[sided])
                    hit_rate = float(correct.mean())
                    z = _hit_zstat(hit_rate, dir_n)

            records.append({
                "signal": signal, "horizon": h, "n": n,
                "direction_hit_rate": hit_rate, "direction_n": dir_n,
                "hit_zstat": z, "ic": ic, "ic_tstat": _ic_tstat(ic, n),
            })

    return pd.DataFrame.from_records(records)


def update_track_record(
    scorecard: pd.DataFrame,
    path: Path | str,
    *,
    as_of: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Append today's scorecard to a dated parquet history and return the full
    history. Running this daily turns single snapshots into a trend you can
    watch (is the edge stable, decaying, or never there?).

    Idempotent per day: re-running for an as_of date replaces that date's rows
    instead of duplicating them.
    """
    path = Path(path)
    stamped = scorecard.copy()
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.utcnow().normalize()
    stamped.insert(0, "as_of", as_of)

    if path.exists():
        try:
            history = pd.read_parquet(path)
            history = history[history["as_of"] != as_of]
        except Exception:
            history = pd.DataFrame()
    else:
        history = pd.DataFrame()
        path.parent.mkdir(parents=True, exist_ok=True)

    out = stamped if history.empty else pd.concat([history, stamped], ignore_index=True)
    out.to_parquet(path, index=False)
    return out
