from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Instrument:
    symbol: str
    name: str
    group: str
    benchmark: str | None = None
    enabled: bool = True


def flatten_watchlist(config: dict) -> list[Instrument]:
    instruments: list[Instrument] = []
    for _, items in config.get("watchlist", {}).items():
        for item in items:
            if item.get("enabled", True):
                instruments.append(
                    Instrument(
                        symbol=item["symbol"],
                        name=item.get("name", item["symbol"]),
                        group=item.get("group", "未分類"),
                        benchmark=item.get("benchmark"),
                        enabled=item.get("enabled", True),
                    )
                )
    return instruments


def symbols(instruments: Iterable[Instrument]) -> list[str]:
    return [instrument.symbol for instrument in instruments]
