from __future__ import annotations

from datetime import date
from pathlib import Path
import pandas as pd

from .market_summary import build_market_summary


def _fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x * 100:.2f}%"


def _fmt_num(x: float) -> str:
    if pd.isna(x):
        return "—"
    return f"{x:,.2f}"


def latest_rows(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty:
        return features
    return (
        features.sort_values(["symbol", "datetime"])
        .groupby("symbol", as_index=False)
        .tail(1)
        .sort_values("symbol")
    )


def _fmt_symbols(symbols: list[str]) -> str:
    return "、".join(symbols) if symbols else "資料不足"


def generate_daily_report(
    features: pd.DataFrame,
    output_path: Path,
    title_date: date | None = None,
    health_report: dict | None = None,
) -> Path:
    title_date = title_date or date.today()
    latest = latest_rows(features)

    lines: list[str] = []
    lines.append(f"# VARnet-lite 每日市場觀察報告")
    lines.append(f"")
    lines.append(f"日期：{title_date.isoformat()}")
    lines.append("")
    lines.append("> 本報告為量化訊號與市場資訊整理，不構成買賣建議。v0.1 只做異常偵測與觀察，不做交易決策。")
    lines.append("")

    if latest.empty:
        lines.append("目前沒有可用資料。")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    summary = build_market_summary(latest, health_report)
    lines.append("## 0. 今日自動摘要")
    lines.append("")
    lines.append(f"- **市場狀態：** {summary['market_state']}")
    lines.append(f"- **平均風險：** {_fmt_num(summary['overall_risk'])}")
    lines.append(f"- **異常標的：** {_fmt_symbols(summary['abnormal_symbols'])}")
    lines.append(f"- **相對強勢：** {_fmt_symbols(summary['strong_symbols'])}")
    lines.append(f"- **相對弱勢：** {_fmt_symbols(summary['weak_symbols'])}")
    lines.append(f"- **資料品質：** {summary['data_health']}")
    lines.append("- **觀察重點：**")
    for point in summary["watch_points"]:
        lines.append(f"  - {point}")
    lines.append("")

    overall_risk = summary["overall_risk"]
    lines.append("## 1. 市場總覽")
    lines.append("")
    lines.append(f"- 平均風險分數：{_fmt_num(overall_risk)}")
    lines.append("- 主要目的：找出相對強弱、波動擴大與異常訊號。")
    lines.append("")

    lines.append("## 2. 標的分數表")
    lines.append("")
    lines.append("| 標的 | 收盤 | 1日報酬 | 20日報酬 | 趨勢 | 動能 | 相對強弱 | 風險 | 狀態 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for _, row in latest.iterrows():
        lines.append(
            "| {symbol} | {close} | {r1} | {r20} | {trend} | {momentum} | {rs} | {risk} | {label}/{risk_level} |".format(
                symbol=row["symbol"],
                close=_fmt_num(row["close"]),
                r1=_fmt_pct(row.get("return_1d")),
                r20=_fmt_pct(row.get("return_20d")),
                trend=_fmt_num(row.get("trend_score")),
                momentum=_fmt_num(row.get("momentum_score")),
                rs=_fmt_num(row.get("relative_strength_score")),
                risk=_fmt_num(row.get("risk_score")),
                label=row.get("condition_label", "—"),
                risk_level=row.get("risk_level", "—"),
            )
        )
    lines.append("")

    lines.append("## 3. 今日異常訊號")
    lines.append("")
    abnormal = latest.sort_values("risk_score", ascending=False).head(3)
    for _, row in abnormal.iterrows():
        lines.append(
            f"- **{row['symbol']}**：風險分數 {_fmt_num(row.get('risk_score'))}，"
            f"波動風險 {_fmt_num(row.get('volatility_risk_score'))}，"
            f"小波異常 {_fmt_num(row.get('wavelet_anomaly_score'))}，"
            f"量比 {_fmt_num(row.get('volume_ratio'))}。"
        )
    lines.append("")

    lines.append("## 4. 相對強弱排序")
    lines.append("")
    lines.append("**相對強勢：** " + _fmt_symbols(summary["strong_symbols"]))
    lines.append("")
    lines.append("**相對弱勢：** " + _fmt_symbols(summary["weak_symbols"]))
    lines.append("")

    lines.append("## 5. 隔日觀察重點")
    lines.append("")
    for point in summary["watch_points"]:
        lines.append(f"- {point}")
    lines.append("")

    lines.append("## 6. v0.2 預留：高低區間預測")
    lines.append("")
    lines.append("訓練與回測資料已預留 1、5、10 個交易日的未來高低區間標籤；每日推論特徵不包含這些欄位，避免偷看未來資料。")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
