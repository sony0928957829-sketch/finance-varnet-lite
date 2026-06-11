from __future__ import annotations

from datetime import date
import json
from urllib.request import Request, urlopen

import pandas as pd

from .base_fetcher import BaseFetcher
from .yfinance_fetcher import YFINANCE_COLUMNS
from src.normalizers.alternative_data import CHIP_COLUMNS


class TwseFetcher(BaseFetcher):
    """TWSE OpenAPI latest daily snapshot fallback.

    The public STOCK_DAY_ALL endpoint is a latest-snapshot source, so it is
    intended for repairing the newest bar rather than five-year backfills.
    """

    source_name = "twse"
    endpoint = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
    institutional_endpoint = "https://openapi.twse.com.tw/v1/fund/T86"
    margin_endpoint = "https://openapi.twse.com.tw/v1/exchangeReport/MI_MARGN"

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
            raise ValueError("TwseFetcher supports daily ('1d') data only.")
        payload = self._fetch_json(self.endpoint)
        as_of = date.fromisoformat(str(end)) if end else date.today()
        return self.normalize_snapshot(pd.DataFrame(payload), symbols, as_of=as_of)

    def fetch_institutional_history(
        self,
        symbols: list[str],
        start: str | date,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        as_of = date.fromisoformat(str(end)) if end else date.today()
        raw = pd.DataFrame(self._fetch_json(self.institutional_endpoint))
        return self.normalize_institutional_snapshot(raw, symbols, as_of=as_of)

    def fetch_margin_short_history(
        self,
        symbols: list[str],
        start: str | date,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        as_of = date.fromisoformat(str(end)) if end else date.today()
        raw = pd.DataFrame(self._fetch_json(self.margin_endpoint))
        return self.normalize_margin_snapshot(raw, symbols, as_of=as_of)

    def normalize_snapshot(
        self,
        raw: pd.DataFrame,
        symbols: list[str],
        *,
        as_of: date,
    ) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame(columns=YFINANCE_COLUMNS)
        wanted = {symbol.removesuffix(".TW"): symbol for symbol in symbols}
        rows = raw[raw["Code"].astype(str).isin(wanted)].copy().reset_index(drop=True)
        if rows.empty:
            return pd.DataFrame(columns=YFINANCE_COLUMNS)
        created_at = pd.Timestamp.now(tz="UTC")
        frame = pd.DataFrame(
            {
                "datetime": pd.Timestamp(as_of),
                "symbol": rows["Code"].astype(str).map(wanted),
                "market": "TW",
                "timeframe": interval_value("1d", len(rows)),
                "open": _numeric(rows, "OpeningPrice"),
                "high": _numeric(rows, "HighestPrice"),
                "low": _numeric(rows, "LowestPrice"),
                "close": _numeric(rows, "ClosingPrice"),
                "volume": _numeric(rows, "TradeVolume"),
                "source": self.source_name,
                "adjusted": False,
                "created_at": created_at,
            }
        )
        return frame[YFINANCE_COLUMNS].dropna(subset=["close"]).reset_index(drop=True)

    def normalize_institutional_snapshot(
        self,
        raw: pd.DataFrame,
        symbols: list[str],
        *,
        as_of: date,
    ) -> pd.DataFrame:
        rows: list[dict] = []
        created_at = pd.Timestamp.now(tz="UTC")
        wanted = {symbol.removesuffix(".TW"): symbol for symbol in symbols}
        for _, item in raw.iterrows():
            code = str(_first_value(item, "證券代號", "Code", "stock_id") or "").strip()
            if code not in wanted:
                continue
            metrics = {
                "foreign_net_buy": _net_value(
                    item,
                    (
                        "外陸資買進股數(不含外資自營商)",
                        "Foreign_Investor_Buy",
                    ),
                    (
                        "外陸資賣出股數(不含外資自營商)",
                        "Foreign_Investor_Sell",
                    ),
                    ("外陸資買賣超股數(不含外資自營商)",),
                ),
                "investment_trust_net_buy": _net_value(
                    item,
                    ("投信買進股數", "Investment_Trust_Buy"),
                    ("投信賣出股數", "Investment_Trust_Sell"),
                    ("投信買賣超股數",),
                ),
                "dealer_net_buy": _net_value(
                    item,
                    ("自營商買進股數(自行買賣)", "Dealer_Buy"),
                    ("自營商賣出股數(自行買賣)", "Dealer_Sell"),
                    ("自營商買賣超股數",),
                ),
            }
            for metric, value in metrics.items():
                if pd.isna(value):
                    continue
                rows.append(
                    {
                        "datetime": pd.Timestamp(as_of),
                        "symbol": wanted[code],
                        "market": "TW",
                        "dataset": "institutional",
                        "metric": metric,
                        "value": value,
                        "unit": "shares",
                        "source": self.source_name,
                        "created_at": created_at,
                    }
                )
        return pd.DataFrame(rows, columns=CHIP_COLUMNS)

    def normalize_margin_snapshot(
        self,
        raw: pd.DataFrame,
        symbols: list[str],
        *,
        as_of: date,
    ) -> pd.DataFrame:
        rows: list[dict] = []
        created_at = pd.Timestamp.now(tz="UTC")
        wanted = {symbol.removesuffix(".TW"): symbol for symbol in symbols}
        for _, item in raw.iterrows():
            code = str(
                _first_value(item, "股票代號", "證券代號", "Code", "stock_id") or ""
            ).strip()
            if code not in wanted:
                continue
            for metric, candidates in {
                "margin_balance": (
                    "融資今日餘額",
                    "MarginPurchaseTodayBalance",
                ),
                "short_balance": (
                    "融券今日餘額",
                    "ShortSaleTodayBalance",
                ),
            }.items():
                value = _numeric_value(_first_value(item, *candidates))
                if pd.isna(value):
                    continue
                rows.append(
                    {
                        "datetime": pd.Timestamp(as_of),
                        "symbol": wanted[code],
                        "market": "TW",
                        "dataset": "margin_short",
                        "metric": metric,
                        "value": value,
                        "unit": "shares",
                        "source": self.source_name,
                        "created_at": created_at,
                    }
                )
        return pd.DataFrame(rows, columns=CHIP_COLUMNS)

    def _fetch_json(self, endpoint: str) -> list[dict]:
        request = Request(endpoint, headers={"User-Agent": "VARnet-lite/0.2"})
        with urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8-sig"))


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(
        frame[column].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )


def interval_value(value: str, size: int) -> pd.Series:
    return pd.Series([value] * size)


def _first_value(item: pd.Series, *candidates: str):
    for candidate in candidates:
        if candidate in item.index and pd.notna(item.get(candidate)):
            return item.get(candidate)
    return None


def _numeric_value(value) -> float:
    if value is None:
        return float("nan")
    cleaned = str(value).replace(",", "").replace("+", "")
    converted = pd.to_numeric(pd.Series([cleaned]), errors="coerce").iloc[0]
    return float(converted) if pd.notna(converted) else float("nan")


def _net_value(
    item: pd.Series,
    buy_candidates: tuple[str, ...],
    sell_candidates: tuple[str, ...],
    net_candidates: tuple[str, ...],
) -> float:
    net = _numeric_value(_first_value(item, *net_candidates))
    if pd.notna(net):
        return net
    buy = _numeric_value(_first_value(item, *buy_candidates))
    sell = _numeric_value(_first_value(item, *sell_candidates))
    if pd.isna(buy) or pd.isna(sell):
        return float("nan")
    return buy - sell
