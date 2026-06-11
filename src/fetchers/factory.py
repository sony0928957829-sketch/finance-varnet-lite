from __future__ import annotations

from .base_fetcher import BaseFetcher
from .mock_fetcher import MockFetcher
from .yfinance_fetcher import YFinanceFetcher
from .finmind_fetcher import FinMindFetcher
from .fred_fetcher import FredFetcher
from .taifex_fetcher import TaifexFetcher
from .twse_fetcher import TwseFetcher


def create_fetcher(
    mode: str,
    *,
    symbol_aliases: dict[str, str] | None = None,
) -> BaseFetcher:
    mode = mode.lower().strip()
    if mode == "mock":
        return MockFetcher()
    if mode == "yfinance":
        return YFinanceFetcher(symbol_aliases=symbol_aliases)
    if mode == "finmind":
        return FinMindFetcher()
    if mode == "twse":
        return TwseFetcher()
    if mode == "taifex":
        return TaifexFetcher()
    if mode == "fred":
        return FredFetcher()
    raise ValueError(
        f"Unsupported mode: {mode}. Use mock, yfinance, finmind, twse, taifex, or fred."
    )
