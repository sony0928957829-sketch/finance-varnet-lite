from __future__ import annotations

"""Human-readable summary of the signal-validation track record.

Reads the accumulating scorecard (one batch of rows per day) produced by
signal_validation.update_track_record and turns it into a short Markdown
verdict per signal: latest hit-rate / IC, statistical significance, whether the
edge is trending up or down, and a plain-language call. It deliberately reports
"insufficient data" instead of a verdict when the sample is thin.

This measures the signals; it is NOT investment advice.
"""

from pathlib import Path

import numpy as np
import pandas as pd

MIN_OBS = 20          # below this, no verdict
SIG_T = 2.0           # |t|/|z| threshold for "significant"


def _verdict(row: pd.Series) -> str:
    n = row.get("n", 0)
    ic = row.get("ic", np.nan)
    ic_t = row.get("ic_tstat", np.nan)
    hit = row.get("direction_hit_rate", np.nan)
    hit_z = row.get("hit_zstat", np.nan)
    if not n or n < MIN_OBS or not np.isfinite(ic):
        return "資料不足"
    ic_sig = np.isfinite(ic_t) and abs(ic_t) >= SIG_T
    hit_sig = np.isfinite(hit_z) and abs(hit_z) >= SIG_T
    positive = (ic > 0) and (not np.isfinite(hit) or hit >= 0.5)
    if ic_sig and positive:
        return "顯著正向 edge"
    if ic_sig and ic < 0:
        return "顯著反向(可反向或排除)"
    if hit_sig and np.isfinite(hit) and hit > 0.5 and ic > 0:
        return "方向有 edge(IC 偏弱)"
    return "無顯著 edge"


def _trend(history: pd.DataFrame, signal: str, horizon) -> str:
    """Is this signal's IC rising or falling across the track record?"""
    sub = history[(history["signal"] == signal) & (history["horizon"] == horizon)]
    sub = sub.dropna(subset=["ic"]).sort_values("as_of")
    if len(sub) < 4:
        return "—"
    half = len(sub) // 2
    early = sub["ic"].iloc[:half].mean()
    late = sub["ic"].iloc[half:].mean()
    delta = late - early
    if delta > 0.02:
        return "↑ 走強"
    if delta < -0.02:
        return "↓ 走弱"
    return "→ 持平"


def build_validation_summary(history: pd.DataFrame, *, as_of=None) -> str:
    if history is None or history.empty:
        return "# 訊號驗證週報\n\n目前還沒有累積到驗證資料(track record 為空)。\n"

    history = history.copy()
    if "as_of" in history.columns:
        history["as_of"] = pd.to_datetime(history["as_of"], errors="coerce")
        latest_date = history["as_of"].max()
    else:
        latest_date = None
    as_of = as_of or latest_date

    # Latest row per (signal, horizon).
    latest = (
        history.sort_values("as_of")
        .groupby(["signal", "horizon"], as_index=False)
        .tail(1)
        .sort_values(["horizon", "signal"])
    )

    days = history["as_of"].nunique() if "as_of" in history.columns else len(latest)
    lines = [
        "# 訊號驗證週報",
        "",
        f"- 統計截至:{pd.Timestamp(as_of).date() if as_of is not None else 'n/a'}",
        f"- 已累積天數:{days}",
        "",
        "> 本表衡量各訊號對「未來報酬」的預測力,**不是買賣建議**。"
        "命中率需顯著高於 50%、IC 的 t 值站得住(|t|≥2),才算有 edge。",
        "",
        "| 訊號 | 期間(日) | 樣本數 | 命中率 | 命中率 z | IC | IC t | 趨勢 | 判定 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def fmt(x, pct=False):
        if not np.isfinite(x):
            return "—"
        return f"{x*100:.1f}%" if pct else f"{x:.3f}"

    for _, r in latest.iterrows():
        lines.append(
            "| {signal} | {h} | {n} | {hit} | {hz} | {ic} | {ict} | {tr} | {v} |".format(
                signal=r["signal"],
                h=int(r["horizon"]),
                n=int(r.get("n", 0) or 0),
                hit=fmt(r.get("direction_hit_rate", np.nan), pct=True),
                hz=fmt(r.get("hit_zstat", np.nan)),
                ic=fmt(r.get("ic", np.nan)),
                ict=fmt(r.get("ic_tstat", np.nan)),
                tr=_trend(history, r["signal"], r["horizon"]),
                v=_verdict(r),
            )
        )

    edges = [r for _, r in latest.iterrows() if _verdict(r) == "顯著正向 edge"]
    lines += ["", "## 一句話結論", ""]
    if edges:
        names = ", ".join(sorted({e["signal"] for e in edges}))
        lines.append(f"目前呈現顯著正向 edge 的訊號:**{names}**。仍須持續觀察穩定性,且這是決策參考、非建議。")
    else:
        lines.append("目前**沒有**任何訊號達到顯著正向 edge。最誠實的解讀:還不足以用來定方向——繼續累積資料再看。")
    lines.append("")
    return "\n".join(lines)


def write_validation_summary(
    track_record_path: Path | str,
    out_path: Path | str,
    *,
    as_of=None,
) -> Path | None:
    """Read the track record parquet and write the Markdown summary. NaN-safe."""
    track_record_path = Path(track_record_path)
    out_path = Path(out_path)
    try:
        history = pd.read_parquet(track_record_path) if track_record_path.exists() else pd.DataFrame()
    except Exception:
        history = pd.DataFrame()
    text = build_validation_summary(history, as_of=as_of)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path
