from __future__ import annotations

from datetime import date
from typing import Callable

import pandas as pd

from .factory import create_fetcher
from .yfinance_fetcher import YFINANCE_COLUMNS


def iter_price_routes(config: dict):
    datasets = config.get("datasets", {})
    for category in ("prices", "macro"):
        for route_name, route in datasets.get(category, {}).items():
            yield category, route_name, route


def fetch_prices_with_fallback(
    config: dict,
    *,
    primary_provider: str,
    start: str | date,
    end: str | date | None,
    interval: str = "1d",
    fetcher_factory: Callable = create_fetcher,
) -> tuple[pd.DataFrame, dict]:
    providers = config.get("providers", {})
    frames: list[pd.DataFrame] = []
    route_status: dict[str, dict] = {}

    for category, route_name, route in iter_price_routes(config):
        if not route.get("enabled") or route.get("primary") != primary_provider:
            continue
        expected = list(dict.fromkeys(route.get("symbols", [])))
        minimum_rows = int(route.get("minimum_rows", 1))
        remaining = expected.copy()
        attempts: list[dict] = []

        for priority, provider in enumerate(
            [route.get("primary"), *route.get("fallback", [])]
        ):
            provider_config = providers.get(provider, {})
            if not provider or not provider_config.get("enabled", False) or not remaining:
                continue
            aliases = route.get("provider_symbols", {}).get(provider, {})
            try:
                fetcher = fetcher_factory(provider, symbol_aliases=aliases)
                fetched = fetcher.fetch_price_history(
                    remaining,
                    start=start,
                    end=end,
                    interval=interval,
                )
                counts = fetched.get("symbol", pd.Series(dtype=str)).value_counts()
                observed = sorted(counts.index.astype(str).tolist())
                sufficient = sorted(
                    symbol for symbol, count in counts.items() if count >= minimum_rows
                )
                if not fetched.empty:
                    fetched = fetched.copy()
                    fetched["_source_priority"] = priority
                    frames.append(fetched)
                    remaining = [symbol for symbol in remaining if symbol not in sufficient]
                attempts.append(
                    {
                        "provider": provider,
                        "status": (
                            "ok"
                            if sufficient
                            else "insufficient"
                            if observed
                            else "empty"
                        ),
                        "symbols": observed,
                        "row_counts": {str(symbol): int(count) for symbol, count in counts.items()},
                    }
                )
            except Exception as exc:
                attempts.append(
                    {
                        "provider": provider,
                        "status": "error",
                        "error": str(exc),
                    }
                )

        route_status[f"{category}.{route_name}"] = {
            "expected_symbols": expected,
            "missing_symbols": remaining,
            "attempts": attempts,
            "status": "ok" if not remaining else "partial" if len(remaining) < len(expected) else "error",
        }

    if not frames:
        return pd.DataFrame(columns=YFINANCE_COLUMNS), route_status
    output = pd.concat(frames, ignore_index=True)
    output = output.sort_values(["symbol", "datetime", "_source_priority"])
    output = output.drop_duplicates(
        subset=["datetime", "symbol", "timeframe"],
        keep="first",
    )
    return output.drop(columns="_source_priority").reset_index(drop=True), route_status
