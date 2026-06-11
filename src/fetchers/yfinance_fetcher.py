from __future__ import annotations

from datetime import date
import pandas as pd

from .base_fetcher import BaseFetcher


YFINANCE_COLUMNS = [
    "datetime",
    "symbol",
    "market",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
    "adjusted",
    "created_at",
]


class YFinanceFetcher(BaseFetcher):
    source_name = "yfinance"

    def __init__(self, symbol_aliases: dict[str, str] | None = None):
        self.symbol_aliases = symbol_aliases or {}

    def fetch_price_history(
        self,
        symbols: list[str],
        start: str | date,
        end: str | date | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        if interval != "1d":
            raise ValueError("YFinanceFetcher currently supports daily ('1d') data only.")

        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("yfinance is not installed. Run: pip install yfinance") from exc

        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            provider_symbol = self.symbol_aliases.get(symbol, symbol)
            data = yf.download(
                provider_symbol,
                start=str(start),
                end=str(end) if end else None,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if data.empty:
                continue
            data = self._flatten_columns(data)
            data = data.reset_index()
            date_col = "Date" if "Date" in data.columns else "Datetime"
            frame = pd.DataFrame(
                {
                    "datetime": pd.to_datetime(data[date_col]),
                    "symbol": symbol,
                    "market": self._infer_market(symbol),
                    "timeframe": interval,
                    "open": data["Open"].astype(float),
                    "high": data["High"].astype(float),
                    "low": data["Low"].astype(float),
                    "close": data["Close"].astype(float),
                    "volume": data.get("Volume", pd.Series(index=data.index, dtype=float)).astype(float),
                    "source": self.source_name,
                    "adjusted": True,
                    "created_at": pd.Timestamp.now(tz="UTC"),
                }
            )
            frames.append(frame[YFINANCE_COLUMNS])
        if not frames:
            return pd.DataFrame(columns=YFINANCE_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _flatten_columns(data: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(data.columns, pd.MultiIndex):
            return data

        price_fields = {"Open", "High", "Low", "Close", "Volume"}
        for level in range(data.columns.nlevels):
            values = set(data.columns.get_level_values(level))
            if price_fields.issubset(values):
                flattened = data.copy()
                flattened.columns = data.columns.get_level_values(level)
                return flattened
        raise ValueError("yfinance response does not contain recognizable OHLCV columns.")

    @staticmethod
    def _infer_market(symbol: str) -> str:
        if symbol == "TAIEX" or symbol.endswith(".TW"):
            return "TW"
        if symbol.endswith("-USD"):
            return "CRYPTO"
        if symbol == "^VIX":
            return "US_VOLATILITY"
        if symbol == "^TNX":
            return "US_RATE"
        if symbol == "DX-Y.NYB":
            return "US_DOLLAR"
        if symbol.endswith("=X"):
            return "FX"
        return "US"
