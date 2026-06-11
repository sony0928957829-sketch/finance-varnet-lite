from __future__ import annotations

from datetime import date
from io import StringIO
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .base_fetcher import BaseFetcher
from .yfinance_fetcher import YFINANCE_COLUMNS


class FredFetcher(BaseFetcher):
    source_name = "fred"
    endpoint = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    symbol_series = {
        "^VIX": "VIXCLS",
        "^TNX": "DGS10",
    }

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def fetch_price_history(
        self,
        symbols: list[str],
        start: str | date,
        end: str | date | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        if interval != "1d":
            raise ValueError("FredFetcher supports daily ('1d') data only.")
        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            series = self.symbol_series.get(symbol)
            if not series:
                continue
            params = {"id": series, "cosd": str(start)}
            if end:
                params["coed"] = str(end)
            request = Request(
                f"{self.endpoint}?{urlencode(params)}",
                headers={"User-Agent": "VARnet-lite/0.2"},
            )
            with urlopen(request, timeout=self.timeout) as response:
                raw = pd.read_csv(StringIO(response.read().decode("utf-8")))
            frames.append(self.normalize_series(raw, symbol=symbol, series=series))
        if not frames:
            return pd.DataFrame(columns=YFINANCE_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def normalize_series(
        self,
        raw: pd.DataFrame,
        *,
        symbol: str,
        series: str,
    ) -> pd.DataFrame:
        value = pd.to_numeric(raw[series], errors="coerce")
        date_column = "DATE" if "DATE" in raw.columns else "observation_date"
        market = "US_VOLATILITY" if symbol == "^VIX" else "US_RATE"
        created_at = pd.Timestamp.now(tz="UTC")
        frame = pd.DataFrame(
            {
                "datetime": pd.to_datetime(raw[date_column], errors="coerce"),
                "symbol": symbol,
                "market": market,
                "timeframe": "1d",
                "open": value,
                "high": value,
                "low": value,
                "close": value,
                "volume": 0.0,
                "source": self.source_name,
                "adjusted": False,
                "created_at": created_at,
            }
        )
        return frame[YFINANCE_COLUMNS].dropna(subset=["datetime", "close"]).reset_index(drop=True)
