from __future__ import annotations

from .base_fetcher import BaseFetcher
from .mock_fetcher import MockFetcher
from .yfinance_fetcher import YFinanceFetcher


def create_fetcher(mode: str) -> BaseFetcher:
    mode = mode.lower().strip()
    if mode == "mock":
        return MockFetcher()
    if mode == "yfinance":
        return YFinanceFetcher()
    raise ValueError(f"Unsupported mode: {mode}. Use 'mock' or 'yfinance'.")
