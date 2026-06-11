from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from src.fetchers.finmind_fetcher import FinMindFetcher
from src.fetchers.news_fetcher import YFinanceNewsFetcher
from src.fetchers.taifex_fetcher import TaifexFetcher
from src.fetchers.twse_fetcher import TwseFetcher
from src.normalizers.alternative_data import (
    CHIP_COLUMNS,
    DERIVATIVE_COLUMNS,
    NEWS_COLUMNS,
    normalize_alternative_frame,
)


CATEGORY_COLUMNS = {
    "news": NEWS_COLUMNS,
    "chip": CHIP_COLUMNS,
    "derivatives": DERIVATIVE_COLUMNS,
}


def collect_supplemental_data(
    config: dict,
    *,
    symbols: list[str],
    start: str | date,
    end: str | date,
    output_dir: Path,
    provider_factory: Callable | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    provider_factory = provider_factory or _provider_adapter
    providers = config.get("providers", {})
    datasets = config.get("datasets", {})
    taiwan_symbols = [symbol for symbol in symbols if symbol.endswith(".TW")]
    status: dict[str, dict] = {}
    collected: dict[str, list[pd.DataFrame]] = {
        "news": [],
        "chip": [],
        "derivatives": [],
    }

    for category in ("news", "chip", "derivatives"):
        for route_name, route in datasets.get(category, {}).items():
            if not route.get("enabled", False):
                continue
            route_start = _bounded_start(
                start,
                end,
                int(route.get("history_days", 30)),
            )
            route_key = f"{category}.{route_name}"
            frame, route_status = _fetch_route_with_fallback(
                category,
                route_name,
                route,
                providers=providers,
                symbols=symbols,
                taiwan_symbols=taiwan_symbols,
                start=route_start,
                end=end,
                provider_factory=provider_factory,
            )
            status[route_key] = route_status
            if frame.empty:
                continue

            columns = CATEGORY_COLUMNS[category]
            normalized = normalize_alternative_frame(frame, columns)
            route_path = output_dir / f"{category}_{route_name}.parquet"
            normalized.to_parquet(route_path, index=False)
            status[route_key]["path"] = str(route_path)
            status[route_key]["rows"] = int(len(normalized))
            if category == "news":
                status[route_key]["latest_titles"] = (
                    normalized.sort_values("datetime", ascending=False)["title"]
                    .dropna()
                    .astype(str)
                    .head(3)
                    .tolist()
                )
            if category == "chip":
                status[route_key]["metrics"] = sorted(
                    normalized["metric"].dropna().astype(str).unique()
                )
            collected[category].append(normalized)

    for category, frames in collected.items():
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True).drop_duplicates()
        combined.to_parquet(output_dir / f"{category}.parquet", index=False)

    return status


def _fetch_route_with_fallback(
    category: str,
    route_name: str,
    route: dict,
    *,
    providers: dict,
    symbols: list[str],
    taiwan_symbols: list[str],
    start: str,
    end: str | date,
    provider_factory: Callable,
) -> tuple[pd.DataFrame, dict]:
    columns = CATEGORY_COLUMNS[category]
    attempts: list[dict] = []
    for provider in [route.get("primary"), *route.get("fallback", [])]:
        provider_config = providers.get(provider, {})
        if not provider or not provider_config.get("enabled", False):
            continue
        try:
            adapter = provider_factory(provider)
            frame = _fetch_provider_route(
                adapter,
                category,
                route_name,
                symbols=symbols,
                taiwan_symbols=taiwan_symbols,
                start=start,
                end=end,
            )
            frame = normalize_alternative_frame(frame, columns)
            attempts.append(
                {
                    "provider": provider,
                    "status": "ok" if not frame.empty else "empty",
                    "rows": int(len(frame)),
                }
            )
            if not frame.empty:
                return frame, {
                    "status": "ok",
                    "provider": provider,
                    "fallback_used": provider != route.get("primary"),
                    "rows": int(len(frame)),
                    "attempts": attempts,
                }
        except Exception as exc:
            attempts.append(
                {
                    "provider": provider,
                    "status": "error",
                    "rows": 0,
                    "error": str(exc),
                }
            )
    return pd.DataFrame(columns=columns), {
        "status": "error" if any(a["status"] == "error" for a in attempts) else "empty",
        "provider": None,
        "fallback_used": False,
        "rows": 0,
        "attempts": attempts,
    }


def _fetch_provider_route(
    adapter,
    category: str,
    route_name: str,
    *,
    symbols: list[str],
    taiwan_symbols: list[str],
    start: str,
    end: str | date,
) -> pd.DataFrame:
    if category == "news":
        provider_symbols = taiwan_symbols if isinstance(adapter, FinMindFetcher) else symbols
        return adapter.fetch_news(provider_symbols, start=start, end=end)
    if category == "chip" and route_name == "taiwan_institutional":
        return adapter.fetch_institutional_history(
            taiwan_symbols,
            start=start,
            end=end,
        )
    if category == "chip" and route_name == "taiwan_margin_short":
        return adapter.fetch_margin_short_history(
            taiwan_symbols,
            start=start,
            end=end,
        )
    if category == "derivatives" and route_name == "taiwan_futures_open_interest":
        return adapter.fetch_futures_open_interest_history(start=start, end=end)
    if category == "derivatives" and route_name == "taiwan_options":
        return adapter.fetch_options_history(start=start, end=end)
    raise ValueError(f"Unsupported supplemental route: {category}.{route_name}")


def _provider_adapter(provider: str):
    if provider == "yahoo_finance_news":
        return YFinanceNewsFetcher()
    if provider == "finmind":
        return FinMindFetcher()
    if provider == "taifex":
        return TaifexFetcher()
    if provider == "twse":
        return TwseFetcher()
    raise ValueError(f"No supplemental adapter for provider: {provider}")


def _bounded_start(
    configured_start: str | date,
    end: str | date,
    history_days: int,
) -> str:
    start_day = date.fromisoformat(str(configured_start))
    end_day = date.fromisoformat(str(end))
    return max(start_day, end_day - timedelta(days=history_days)).isoformat()
