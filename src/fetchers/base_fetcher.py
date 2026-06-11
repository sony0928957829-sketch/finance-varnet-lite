from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
import pandas as pd


class BaseFetcher(ABC):
    """Base interface for all data fetchers.

    A fetcher returns raw or semi-normalized OHLCV data.
    Each new data source should implement this interface and keep source-specific
    logic out of the feature/scoring modules.
    """

    source_name: str

    @abstractmethod
    def fetch_price_history(
        self,
        symbols: list[str],
        start: str | date,
        end: str | date | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        raise NotImplementedError
