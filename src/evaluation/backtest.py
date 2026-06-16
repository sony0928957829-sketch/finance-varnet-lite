from __future__ import annotations

"""Cost-aware top-k backtest.

Turns a predictive signal into a tradeable result: each rebalance it ranks
symbols by the signal, holds an equal-weight basket of the top-k (optionally
short the bottom-k) for `horizon` trading days, and charges a transaction cost
on turnover. It reports annualized return, volatility, Sharpe, max drawdown and
hit rate, compared against an equal-weight buy-and-hold benchmark.

IC tells you a signal is predictive; this tells you whether, after costs, acting
on it would have made money. It is a historical study, NOT investment advice.

Rebalances are non-overlapping (step = horizon) and strictly causal: the signal
at date t selects the basket, realized returns over t..t+h are the payoff.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return float("nan")
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1.0).min())


def run_backtest(
    frame: pd.DataFrame,
    signal_col: str,
    *,
    horizon: int = 5,
    top_k: int = 3,
    cost_bps: float = 10.0,
    price_col: str = "close",
    long_short: bool = False,
) -> dict:
    """Return backtest metrics for `signal_col`. Degrades to NaNs if unusable."""
    empty = {
        "signal": signal_col, "horizon": horizon, "n_periods": 0,
        "total_return": float("nan"), "annualized_return": float("nan"),
        "annualized_vol": float("nan"), "sharpe": float("nan"),
        "max_drawdown": float("nan"), "hit_rate": float("nan"),
        "benchmark_annualized_return": float("nan"), "excess_annualized_return": float("nan"),
    }
    needed = {"symbol", "datetime", price_col, signal_col}
    if not needed.issubset(frame.columns) or frame.empty:
        return empty

    df = frame[["symbol", "datetime", price_col, signal_col]].copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])
    dates = sorted(df["datetime"].unique())
    if len(dates) <= horizon + 1:
        return empty

    close = df.pivot_table(index="datetime", columns="symbol", values=price_col, aggfunc="last").reindex(dates)
    sig = df.pivot_table(index="datetime", columns="symbol", values=signal_col, aggfunc="last").reindex(dates)

    cost = cost_bps / 1e4
    strat_rets, bench_rets = [], []
    for i in range(0, len(dates) - horizon, horizon):
        entry, exit_ = close.iloc[i], close.iloc[i + horizon]
        fwd = (exit_ / entry - 1.0).replace([np.inf, -np.inf], np.nan)
        s = sig.iloc[i].dropna()
        s = s[s.index.isin(fwd.dropna().index)]
        if len(s) < max(2, top_k):
            continue
        longs = s.sort_values(ascending=False).head(top_k).index
        long_ret = fwd[longs].dropna()
        if long_ret.empty:
            continue
        ret = long_ret.mean()
        if long_short and len(s) >= 2 * top_k:
            shorts = s.sort_values().head(top_k).index
            short_ret = fwd[shorts].dropna()
            if not short_ret.empty:
                ret = 0.5 * long_ret.mean() - 0.5 * short_ret.mean()
        ret -= 2 * cost  # round-trip cost on the book each rebalance
        strat_rets.append(float(ret))
        bench_rets.append(float(fwd.dropna().mean()))  # equal-weight buy-and-hold

    if not strat_rets:
        return empty

    strat = np.array(strat_rets)
    bench = np.array(bench_rets)
    periods_per_year = 252.0 / horizon
    span_days = max((pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days, 1)
    years = span_days / 365.25

    def annualize(rets: np.ndarray) -> float:
        equity = np.cumprod(1.0 + rets)
        total = equity[-1] - 1.0
        return float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 else float("nan")

    equity = np.cumprod(1.0 + strat)
    total_return = float(equity[-1] - 1.0)
    ann_ret = annualize(strat)
    ann_vol = float(strat.std(ddof=1) * np.sqrt(periods_per_year)) if len(strat) > 1 else float("nan")
    sharpe = float(ann_ret / ann_vol) if ann_vol and np.isfinite(ann_vol) and ann_vol > 0 else float("nan")
    bench_ann = annualize(bench)

    return {
        "signal": signal_col,
        "horizon": horizon,
        "n_periods": len(strat),
        "total_return": total_return,
        "annualized_return": ann_ret,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(equity),
        "hit_rate": float((strat > 0).mean()),
        "benchmark_annualized_return": bench_ann,
        "excess_annualized_return": (ann_ret - bench_ann) if np.isfinite(ann_ret) and np.isfinite(bench_ann) else float("nan"),
    }


def build_backtest_report(results: list[dict], *, cost_bps: float = 10.0, top_k: int = 3) -> str:
    lines = [
        "# 策略回測(含交易成本)",
        "",
        f"- 設定:每次取前 {top_k} 強、持有至下次再平衡、來回成本 {cost_bps:.0f} bps",
        "",
        "> 這是歷史回測,回答「扣成本後照訊號做會不會賺」,**不是買賣建議**。"
        "重點看是否穩定為正、且**贏過買進持有(超額報酬>0)**。",
        "",
        "| 訊號 | 期數 | 年化報酬 | 年化波動 | Sharpe | 最大回撤 | 勝率 | 買進持有年化 | 超額年化 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]

    def p(x):
        return "—" if not np.isfinite(x) else f"{x*100:.1f}%"

    def f(x):
        return "—" if not np.isfinite(x) else f"{x:.2f}"

    for r in results:
        lines.append(
            "| {s} | {n} | {ar} | {av} | {sh} | {md} | {hr} | {br} | {ex} |".format(
                s=r["signal"], n=r["n_periods"], ar=p(r["annualized_return"]),
                av=p(r["annualized_vol"]), sh=f(r["sharpe"]), md=p(r["max_drawdown"]),
                hr=p(r["hit_rate"]), br=p(r["benchmark_annualized_return"]),
                ex=p(r["excess_annualized_return"]),
            )
        )
    lines += ["", "## 一句話結論", ""]
    winners = [r for r in results if np.isfinite(r["excess_annualized_return"]) and r["excess_annualized_return"] > 0 and r["n_periods"] >= 10]
    if winners:
        names = ", ".join(sorted({w["signal"] for w in winners}))
        lines.append(f"扣成本後贏過買進持有的訊號:**{names}**。仍須留意期數是否足夠、是否跨期穩定;這是參考,非建議。")
    else:
        lines.append("扣成本後**沒有**訊號穩定贏過買進持有。最誠實的解讀:現階段照訊號交易不優於單純持有。")
    lines.append("")
    return "\n".join(lines)


def write_backtest_report(results: list[dict], out_path: Path | str, **kwargs) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_backtest_report(results, **kwargs), encoding="utf-8")
    return out_path
