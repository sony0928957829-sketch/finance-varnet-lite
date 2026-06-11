from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path
from typing import Any

import pandas as pd


class DataHealthError(RuntimeError):
    """Raised when price data is unsafe for report generation."""


def _issue(code: str, severity: str, message: str, symbol: str | None = None) -> dict[str, Any]:
    issue: dict[str, Any] = {
        "code": code,
        "severity": severity,
        "message": message,
    }
    if symbol is not None:
        issue["symbol"] = symbol
    return issue


def _threshold(config: dict[str, Any], section: str, market: str) -> int:
    values = config.get(section, {})
    return int(values.get(market, values.get("default", 4)))


def evaluate_price_health(
    frame: pd.DataFrame,
    *,
    expected_symbols: list[str],
    as_of: date,
    primary_source: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate OHLCV data before features or reports are generated."""
    fail_on = set(config.get("fail_on", []))
    issues: list[dict[str, Any]] = []
    symbol_health: dict[str, dict[str, Any]] = {}

    def add_issue(code: str, message: str, symbol: str | None = None) -> None:
        severity = "error" if code in fail_on else "warning"
        issues.append(_issue(code, severity, message, symbol))

    if frame.empty:
        add_issue("empty_data", "Fetcher returned no price rows.")
    required_columns = set(config.get("required_columns", []))
    missing_columns = sorted(required_columns - set(frame.columns))
    if missing_columns:
        add_issue("missing_columns", f"Missing required columns: {missing_columns}")

    if not frame.empty and not missing_columns:
        data = frame.copy()
        data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce").dt.tz_localize(None)
        price_columns = ["open", "high", "low", "close"]
        for column in [*price_columns, "volume"]:
            data[column] = pd.to_numeric(data[column], errors="coerce")

        minimum_rows = int(config.get("minimum_rows_per_symbol", 2))
        for symbol in expected_symbols:
            rows = data[data["symbol"].eq(symbol)].sort_values("datetime").copy()
            if rows.empty:
                add_issue("missing_symbol", "Expected symbol has no rows.", symbol)
                symbol_health[symbol] = {
                    "row_count": 0,
                    "status": "error",
                    "sources": [],
                }
                continue

            market = str(rows["market"].dropna().iloc[-1]) if rows["market"].notna().any() else "UNKNOWN"
            valid_dates = rows["datetime"].dropna()
            latest = valid_dates.max() if not valid_dates.empty else pd.NaT
            sources = sorted(rows["source"].dropna().astype(str).unique().tolist())
            duplicate_count = int(
                rows.duplicated(subset=["datetime", "symbol", "timeframe"], keep=False).sum()
            )
            null_datetime_count = int(rows["datetime"].isna().sum())
            null_price_count = int(rows[price_columns].isna().any(axis=1).sum())
            non_positive_count = int(rows[price_columns].le(0).any(axis=1).sum())
            invalid_ohlc_count = int(
                (
                    (rows["high"] < rows["low"])
                    | (rows["high"] < rows[["open", "close"]].max(axis=1))
                    | (rows["low"] > rows[["open", "close"]].min(axis=1))
                ).sum()
            )
            negative_volume_count = int(rows["volume"].lt(0).sum())
            null_volume_count = int(rows["volume"].isna().sum())

            if len(rows) < minimum_rows:
                add_issue(
                    "insufficient_rows",
                    f"Only {len(rows)} rows; minimum is {minimum_rows}.",
                    symbol,
                )
            if null_datetime_count:
                add_issue(
                    "invalid_datetime",
                    f"{null_datetime_count} rows have invalid datetimes.",
                    symbol,
                )
            if null_price_count:
                add_issue(
                    "null_price",
                    f"{null_price_count} rows have missing OHLC values.",
                    symbol,
                )
            if non_positive_count:
                add_issue(
                    "non_positive_price",
                    f"{non_positive_count} rows have zero or negative OHLC values.",
                    symbol,
                )
            if invalid_ohlc_count:
                add_issue(
                    "invalid_ohlc",
                    f"{invalid_ohlc_count} rows violate OHLC ordering.",
                    symbol,
                )
            if duplicate_count:
                add_issue(
                    "duplicate_bar",
                    f"{duplicate_count} duplicate bars were found.",
                    symbol,
                )
            if negative_volume_count:
                add_issue(
                    "negative_volume",
                    f"{negative_volume_count} rows have negative volume.",
                    symbol,
                )
            if null_volume_count:
                add_issue(
                    "null_volume",
                    f"{null_volume_count} rows have missing volume.",
                    symbol,
                )

            age_days: int | None = None
            if pd.isna(latest):
                add_issue("invalid_datetime", "No valid datetime is available.", symbol)
            else:
                age_days = (as_of - latest.date()).days
                if age_days < 0:
                    add_issue(
                        "future_datetime",
                        f"Latest bar {latest.date()} is after as-of date {as_of}.",
                        symbol,
                    )
                freshness_limit = _threshold(config, "freshness_max_age_days", market)
                if age_days > freshness_limit:
                    add_issue(
                        "stale_data",
                        f"Latest bar is {age_days} days old; limit is {freshness_limit}.",
                        symbol,
                    )

            gap_limit = _threshold(config, "long_gap_days", market)
            gaps = valid_dates.sort_values().drop_duplicates().diff().dt.days.dropna()
            long_gap_count = int(gaps.gt(gap_limit).sum())
            max_gap_days = int(gaps.max()) if not gaps.empty else 0
            if long_gap_count:
                add_issue(
                    "long_gap",
                    f"{long_gap_count} gaps exceed {gap_limit} days; maximum is {max_gap_days}.",
                    symbol,
                )

            fallback_used = any(source != primary_source for source in sources)
            if fallback_used:
                add_issue(
                    "fallback_source",
                    f"Observed sources {sources}; expected primary source is {primary_source}.",
                    symbol,
                )

            symbol_issue_severities = [
                issue["severity"] for issue in issues if issue.get("symbol") == symbol
            ]
            symbol_health[symbol] = {
                "market": market,
                "row_count": int(len(rows)),
                "first_datetime": valid_dates.min().isoformat() if not valid_dates.empty else None,
                "latest_datetime": latest.isoformat() if not pd.isna(latest) else None,
                "age_days": age_days,
                "sources": sources,
                "fallback_used": fallback_used,
                "duplicate_bar_count": duplicate_count,
                "null_price_count": null_price_count,
                "non_positive_price_count": non_positive_count,
                "invalid_ohlc_count": invalid_ohlc_count,
                "null_volume_count": null_volume_count,
                "negative_volume_count": negative_volume_count,
                "long_gap_count": long_gap_count,
                "max_gap_days": max_gap_days,
                "status": (
                    "error"
                    if "error" in symbol_issue_severities
                    else "warning"
                    if "warning" in symbol_issue_severities
                    else "healthy"
                ),
            }

    error_count = sum(issue["severity"] == "error" for issue in issues)
    warning_count = sum(issue["severity"] == "warning" for issue in issues)
    status = "error" if error_count else "warning" if warning_count else "healthy"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "as_of": as_of.isoformat(),
        "primary_source": primary_source,
        "status": status,
        "summary": {
            "expected_symbol_count": len(expected_symbols),
            "observed_symbol_count": sum(
                details.get("row_count", 0) > 0 for details in symbol_health.values()
            ),
            "error_count": error_count,
            "warning_count": warning_count,
        },
        "symbols": symbol_health,
        "issues": issues,
    }


def write_health_report(report: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def raise_for_health_errors(report: dict[str, Any]) -> None:
    if report.get("status") != "error":
        return
    errors = [
        issue["message"]
        for issue in report.get("issues", [])
        if issue.get("severity") == "error"
    ]
    raise DataHealthError("Price data health check failed: " + "; ".join(errors))
