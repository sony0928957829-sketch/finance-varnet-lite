from __future__ import annotations

from datetime import date, timedelta
import json
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from .base_fetcher import BaseFetcher
from .yfinance_fetcher import YFINANCE_COLUMNS
from src.normalizers.alternative_data import CHIP_COLUMNS, DERIVATIVE_COLUMNS, NEWS_COLUMNS


class FinMindFetcher(BaseFetcher):
    source_name = "finmind"
    base_url = "https://api.finmindtrade.com/api/v4/data"

    def __init__(self, token: str | None = None, timeout: int = 30):
        self.token = token if token is not None else os.getenv("FINMIND_TOKEN")
        self.timeout = timeout

    def fetch_dataset(
        self,
        dataset: str,
        *,
        data_id: str | None = None,
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        params = {"dataset": dataset}
        if data_id:
            params["data_id"] = data_id
        if start:
            params["start_date"] = str(start)
        if end:
            params["end_date"] = str(end)

        headers = {"User-Agent": "VARnet-lite/0.2"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(f"{self.base_url}?{urlencode(params)}", headers=headers)
        with urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if payload.get("status") != 200:
            raise RuntimeError(f"FinMind {dataset} failed: {payload.get('msg', 'unknown error')}")
        return pd.DataFrame(payload.get("data", []))

    def fetch_price_history(
        self,
        symbols: list[str],
        start: str | date,
        end: str | date | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        if interval != "1d":
            raise ValueError("FinMindFetcher supports daily ('1d') data only.")

        frames: list[pd.DataFrame] = []
        for symbol in symbols:
            if symbol == "TWD=X":
                raw = self.fetch_dataset(
                    "TaiwanExchangeRate",
                    data_id="USD",
                    start=start,
                    end=end,
                )
                if raw.empty:
                    continue
                spot_buy = pd.to_numeric(raw["spot_buy"], errors="coerce")
                spot_sell = pd.to_numeric(raw["spot_sell"], errors="coerce")
                price = pd.concat([spot_buy, spot_sell], axis=1).mean(axis=1)
                frame = pd.DataFrame(
                    {
                        "datetime": pd.to_datetime(raw["date"]),
                        "symbol": symbol,
                        "market": "FX",
                        "timeframe": interval,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": 0.0,
                        "source": self.source_name,
                        "adjusted": False,
                        "created_at": pd.Timestamp.now(tz="UTC"),
                    }
                )
            elif symbol == "TX":
                raw = self.fetch_dataset(
                    "TaiwanFuturesDaily",
                    data_id="TX",
                    start=start,
                    end=end,
                )
                if raw.empty:
                    continue
                volume_column = _first_column(raw, "Trading_Volume", "volume")
                raw["_volume"] = pd.to_numeric(raw[volume_column], errors="coerce")
                raw = raw.sort_values("_volume").groupby("date", as_index=False).tail(1)
                frame = pd.DataFrame(
                    {
                        "datetime": pd.to_datetime(raw["date"]),
                        "symbol": symbol,
                        "market": "TW_DERIVATIVES",
                        "timeframe": interval,
                        "open": _numeric_series(raw, "open"),
                        "high": _numeric_series(raw, "max", "high"),
                        "low": _numeric_series(raw, "min", "low"),
                        "close": _numeric_series(raw, "close", "settlement_price"),
                        "volume": raw["_volume"],
                        "source": self.source_name,
                        "adjusted": False,
                        "created_at": pd.Timestamp.now(tz="UTC"),
                    }
                )
            elif symbol == "TAIEX":
                raw = self.fetch_dataset(
                    "TaiwanStockTotalReturnIndex",
                    data_id="TAIEX",
                    start=start,
                    end=end,
                )
                if raw.empty:
                    continue
                price = pd.to_numeric(raw["price"], errors="coerce")
                frame = pd.DataFrame(
                    {
                        "datetime": pd.to_datetime(raw["date"]),
                        "symbol": symbol,
                        "market": "TW",
                        "timeframe": interval,
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": 0.0,
                        "source": self.source_name,
                        "adjusted": True,
                        "created_at": pd.Timestamp.now(tz="UTC"),
                    }
                )
            else:
                data_id = symbol.removesuffix(".TW")
                raw = self.fetch_dataset(
                    "TaiwanStockPriceAdj",
                    data_id=data_id,
                    start=start,
                    end=end,
                )
                if raw.empty:
                    continue
                frame = pd.DataFrame(
                    {
                        "datetime": pd.to_datetime(raw["date"]),
                        "symbol": symbol,
                        "market": "TW",
                        "timeframe": interval,
                        "open": pd.to_numeric(raw["open"], errors="coerce"),
                        "high": pd.to_numeric(raw["max"], errors="coerce"),
                        "low": pd.to_numeric(raw["min"], errors="coerce"),
                        "close": pd.to_numeric(raw["close"], errors="coerce"),
                        "volume": pd.to_numeric(raw["Trading_Volume"], errors="coerce"),
                        "source": self.source_name,
                        "adjusted": True,
                        "created_at": pd.Timestamp.now(tz="UTC"),
                    }
                )
            frames.append(frame[YFINANCE_COLUMNS])
        if not frames:
            return pd.DataFrame(columns=YFINANCE_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def fetch_institutional_history(
        self,
        symbols: list[str],
        start: str | date,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        rows: list[dict] = []
        created_at = pd.Timestamp.now(tz="UTC")
        institution_names = {
            "Foreign_Investor": "foreign_net_buy",
            "Investment_Trust": "investment_trust_net_buy",
            "Dealer_self": "dealer_net_buy",
            "Dealer_Hedging": "dealer_hedging_net_buy",
        }
        for symbol in symbols:
            raw = self.fetch_dataset(
                "TaiwanStockInstitutionalInvestorsBuySell",
                data_id=symbol.removesuffix(".TW"),
                start=start,
                end=end,
            )
            for _, item in raw.iterrows():
                rows.append(
                    {
                        "datetime": item.get("date"),
                        "symbol": symbol,
                        "market": "TW",
                        "dataset": "institutional",
                        "metric": institution_names.get(
                            str(item.get("name")),
                            str(item.get("name")),
                        ),
                        "value": _number(item.get("buy")) - _number(item.get("sell")),
                        "unit": "shares",
                        "source": self.source_name,
                        "created_at": created_at,
                    }
                )
        return pd.DataFrame(rows, columns=CHIP_COLUMNS)

    def fetch_margin_short_history(
        self,
        symbols: list[str],
        start: str | date,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        rows: list[dict] = []
        created_at = pd.Timestamp.now(tz="UTC")
        for symbol in symbols:
            raw = self.fetch_dataset(
                "TaiwanStockMarginPurchaseShortSale",
                data_id=symbol.removesuffix(".TW"),
                start=start,
                end=end,
            )
            for _, item in raw.iterrows():
                for field, metric in {
                    "MarginPurchaseTodayBalance": "margin_balance",
                    "ShortSaleTodayBalance": "short_balance",
                }.items():
                    rows.append(
                        {
                            "datetime": item.get("date"),
                            "symbol": symbol,
                            "market": "TW",
                            "dataset": "margin_short",
                            "metric": metric,
                            "value": _number(item.get(field)),
                            "unit": "shares",
                            "source": self.source_name,
                            "created_at": created_at,
                        }
                    )
        return pd.DataFrame(rows, columns=CHIP_COLUMNS)

    def fetch_futures_open_interest_history(
        self,
        start: str | date,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        return self._fetch_derivative_dataset(
            "TaiwanFuturesDaily",
            "TX",
            start=start,
            end=end,
        )

    def fetch_options_history(
        self,
        start: str | date,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        raw = self.fetch_dataset(
            "TaiwanOptionDaily",
            data_id="TXO",
            start=start,
            end=end,
        )
        raw_rows = self._fetch_derivative_dataset(
            "TaiwanOptionDaily",
            "TXO",
            start=start,
            end=end,
            raw=raw,
        )
        ratios = _option_ratio_rows(raw, pd.Timestamp.now(tz="UTC"))
        return pd.concat([raw_rows, ratios], ignore_index=True)

    def _fetch_derivative_dataset(
        self,
        dataset: str,
        symbol: str,
        *,
        start: str | date,
        end: str | date | None,
        raw: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        if raw is None:
            raw = self.fetch_dataset(dataset, data_id=symbol, start=start, end=end)
        created_at = pd.Timestamp.now(tz="UTC")
        rows: list[dict] = []
        for _, item in raw.iterrows():
            rows.append(
                {
                    "datetime": item.get("date"),
                    "symbol": symbol,
                    "market": "TW_DERIVATIVES",
                    "dataset": dataset,
                    "contract": item.get("futures_id", item.get("option_id", symbol)),
                    "expiry": item.get("contract_date", item.get("delivery_month")),
                    "option_type": item.get("call_put", item.get("option_type")),
                    "strike": _number(item.get("strike_price")),
                    "open": _number(item.get("open")),
                    "high": _number(item.get("max")),
                    "low": _number(item.get("min")),
                    "close": _number(item.get("close")),
                    "settlement": _number(item.get("settlement_price")),
                    "volume": _number(item.get("Trading_Volume", item.get("volume"))),
                    "open_interest": _number(item.get("open_interest")),
                    "value": float("nan"),
                    "source": self.source_name,
                    "created_at": created_at,
                }
            )
        return pd.DataFrame(rows, columns=DERIVATIVE_COLUMNS)

    def fetch_news(
        self,
        symbols: list[str],
        start: str | date,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        rows: list[dict] = []
        created_at = pd.Timestamp.now(tz="UTC")
        start_day = date.fromisoformat(str(start))
        end_day = date.fromisoformat(str(end)) if end else start_day
        for symbol in symbols:
            day = start_day
            while day <= end_day:
                raw = self.fetch_dataset(
                    "TaiwanStockNews",
                    data_id=symbol.removesuffix(".TW"),
                    start=day,
                )
                for _, item in raw.iterrows():
                    rows.append(
                        {
                            "datetime": item.get("date"),
                            "symbol": symbol,
                            "market": "TW",
                            "event_type": "news",
                            "title": item.get("title", ""),
                            "summary": item.get(
                                "description",
                                item.get("summary", ""),
                            ),
                            "url": item.get("link", item.get("url", "")),
                            "publisher": item.get("source", ""),
                            "source": self.source_name,
                            "created_at": created_at,
                        }
                    )
                day += timedelta(days=1)
        return pd.DataFrame(rows, columns=NEWS_COLUMNS)


def _number(value) -> float:
    converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(converted) if pd.notna(converted) else float("nan")


def _first_column(frame: pd.DataFrame, *names: str) -> str:
    for name in names:
        if name in frame.columns:
            return name
    raise ValueError(f"FinMind response is missing columns: {names}")


def _numeric_series(frame: pd.DataFrame, *names: str) -> pd.Series:
    return pd.to_numeric(frame[_first_column(frame, *names)], errors="coerce")


def _optional_column(frame: pd.DataFrame, *names: str) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def _ratio_row(trading_day, dataset: str, value: float, created_at) -> dict:
    return {
        "datetime": trading_day,
        "symbol": "TXO",
        "market": "TW_DERIVATIVES",
        "dataset": dataset,
        "contract": "TXO",
        "expiry": None,
        "option_type": None,
        "strike": float("nan"),
        "open": float("nan"),
        "high": float("nan"),
        "low": float("nan"),
        "close": float("nan"),
        "settlement": float("nan"),
        "volume": float("nan"),
        "open_interest": float("nan"),
        "value": value,
        "source": "finmind",
        "created_at": created_at,
    }


def _option_ratio_rows(raw: pd.DataFrame, created_at) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=DERIVATIVE_COLUMNS)
    option_type = _optional_column(raw, "call_put", "option_type")
    if not option_type:
        return pd.DataFrame(columns=DERIVATIVE_COLUMNS)

    working = raw.assign(_type=raw[option_type].astype(str).str.lower())
    rows: list[dict] = []
    for value_column, dataset in [
        (_optional_column(raw, "Trading_Volume", "volume"), "put_call_volume_ratio"),
        (_optional_column(raw, "open_interest"), "put_call_open_interest_ratio"),
    ]:
        if not value_column:
            continue
        working["_value"] = pd.to_numeric(raw[value_column], errors="coerce")
        for trading_day, day_rows in working.groupby("date"):
            call_value = day_rows.loc[
                day_rows["_type"].str.contains("call|買權", regex=True),
                "_value",
            ].sum()
            put_value = day_rows.loc[
                day_rows["_type"].str.contains("put|賣權", regex=True),
                "_value",
            ].sum()
            rows.append(
                _ratio_row(
                    trading_day,
                    dataset,
                    put_value / call_value if call_value else float("nan"),
                    created_at,
                )
            )
    return pd.DataFrame(rows, columns=DERIVATIVE_COLUMNS)
