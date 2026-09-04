from __future__ import annotations

from datetime import date
import importlib.util
import time

import pandas as pd

from .base_fetcher import BaseFetcher


# yfinance's `repair=True` path reconstructs suspicious bars and imports
# scikit-learn (yfinance/scrapers/history.py -> "from sklearn.cluster import
# DBSCAN"). scikit-learn is not a declared yfinance dependency, so in an
# environment without it EVERY download fails with ModuleNotFoundError and the
# pipeline sees an empty market. requirements.txt now pins scikit-learn, and
# this probe is the belt-and-braces: if it is ever missing again, we downgrade
# to repair=False and still return prices instead of losing the whole run.
def sklearn_available() -> bool:
    try:
        return importlib.util.find_spec("sklearn") is not None
    except (ImportError, ValueError):  # pragma: no cover - defensive
        return False


# Transient Yahoo throttling (HTTP 429 / empty payloads) is common from CI
# runners, so a symbol is retried a few times before it is declared missing.
DOWNLOAD_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 2.0


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
            data = self._download(
                yf,
                provider_symbol,
                start=start,
                end=end,
                interval=interval,
            )
            if data is None or data.empty:
                print(f"[warn] yfinance returned no rows for {symbol} after retries.")
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
            frame = self._enforce_ohlc_bounds(frame)
            frames.append(frame[YFINANCE_COLUMNS])
        if not frames:
            return pd.DataFrame(columns=YFINANCE_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _download(
        yf,
        provider_symbol: str,
        *,
        start: str | date,
        end: str | date | None,
        interval: str,
    ) -> pd.DataFrame | None:
        """Download one symbol, degrading instead of failing.

        Order of attempts, from best data quality to most permissive:
        1. repair=True (needs scikit-learn) with retries for transient errors;
        2. repair=False, which never touches the sklearn code path.

        Returning None/empty is a per-symbol outcome; the caller decides whether
        the run still has enough coverage to be worth reporting.
        """
        repair_modes = [True, False] if sklearn_available() else [False]
        if not sklearn_available():
            print(
                "[warn] scikit-learn is missing; yfinance price repair is disabled. "
                "Install scikit-learn (see requirements.txt) for repaired bars."
            )

        for repair in repair_modes:
            for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
                try:
                    data = yf.download(
                        provider_symbol,
                        start=str(start),
                        end=str(end) if end else None,
                        interval=interval,
                        auto_adjust=True,
                        repair=repair,
                        progress=False,
                        threads=False,
                    )
                except Exception as exc:  # network / upstream API changes
                    print(
                        f"[warn] yfinance download failed for {provider_symbol} "
                        f"(repair={repair}, attempt {attempt}/{DOWNLOAD_ATTEMPTS}): {exc}"
                    )
                    data = None
                if data is not None and not data.empty:
                    return data
                if attempt < DOWNLOAD_ATTEMPTS:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        return None

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
    def _enforce_ohlc_bounds(frame: pd.DataFrame) -> pd.DataFrame:
        """Repair provider bars whose high/low do not contain open and close."""
        output = frame.copy()
        price_columns = ["open", "high", "low", "close"]
        numeric = output[price_columns].apply(pd.to_numeric, errors="coerce")
        output["high"] = numeric.max(axis=1, skipna=False)
        output["low"] = numeric.min(axis=1, skipna=False)
        return output

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
