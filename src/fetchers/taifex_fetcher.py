from __future__ import annotations

from datetime import date, timedelta
from html.parser import HTMLParser
from io import BytesIO, StringIO
import re
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd

from .base_fetcher import BaseFetcher
from .yfinance_fetcher import YFINANCE_COLUMNS
from src.normalizers.alternative_data import DERIVATIVE_COLUMNS


class TaifexFetcher(BaseFetcher):
    source_name = "taifex"
    daily_url = (
        "https://www.taifex.com.tw/file/taifex/"
        "Dailydownload/DailydownloadCSV/Daily_{date}.zip"
    )

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
            raise ValueError("TaifexFetcher supports daily ('1d') data only.")
        if any(symbol != "TX" for symbol in symbols):
            raise ValueError("TaifexFetcher currently supports TX only.")

        start_day = date.fromisoformat(str(start))
        end_day = date.fromisoformat(str(end)) if end else date.today()
        start_day = max(start_day, end_day - timedelta(days=10))
        frames: list[pd.DataFrame] = []
        day = start_day
        while day <= end_day:
            if day.weekday() < 5:
                try:
                    frames.append(self.fetch_daily_tx(day))
                except Exception:
                    pass
            day += timedelta(days=1)
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame(columns=YFINANCE_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def fetch_daily_tx(self, trading_day: date) -> pd.DataFrame:
        url = self.daily_url.format(date=trading_day.strftime("%Y_%m_%d"))
        request = Request(url, headers={"User-Agent": "VARnet-lite/0.2"})
        with urlopen(request, timeout=self.timeout) as response:
            archive = ZipFile(BytesIO(response.read()))
        csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        content = archive.read(csv_name)
        return self.parse_futures_csv(content)

    def fetch_futures_open_interest_history(
        self,
        start: str | date,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        start_day = date.fromisoformat(str(start))
        end_day = date.fromisoformat(str(end)) if end else date.today()
        start_day = max(start_day, end_day - timedelta(days=10))
        frames: list[pd.DataFrame] = []
        day = start_day
        while day <= end_day:
            if day.weekday() < 5:
                try:
                    frames.append(self.fetch_daily_tx_derivatives(day))
                except Exception:
                    pass
            day += timedelta(days=1)
        frames = [frame for frame in frames if not frame.empty]
        if not frames:
            return pd.DataFrame(columns=DERIVATIVE_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def fetch_options_history(
        self,
        start: str | date,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        request = Request(
            "https://www.taifex.com.tw/cht/3/pcRatio",
            headers={"User-Agent": "VARnet-lite/0.2"},
        )
        with urlopen(request, timeout=self.timeout) as response:
            html = response.read().decode("utf-8", errors="replace")
        frame = self.parse_put_call_ratio_html(html)
        if frame.empty:
            return frame
        start_day = pd.Timestamp(start)
        end_day = pd.Timestamp(end) if end else pd.Timestamp.today()
        datetimes = pd.to_datetime(frame["datetime"], errors="coerce")
        return frame[datetimes.between(start_day, end_day)].reset_index(drop=True)

    def fetch_daily_tx_derivatives(self, trading_day: date) -> pd.DataFrame:
        url = self.daily_url.format(date=trading_day.strftime("%Y_%m_%d"))
        request = Request(url, headers={"User-Agent": "VARnet-lite/0.2"})
        with urlopen(request, timeout=self.timeout) as response:
            archive = ZipFile(BytesIO(response.read()))
        csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        return self.parse_futures_derivatives_csv(archive.read(csv_name))

    def parse_futures_csv(self, content: bytes | str) -> pd.DataFrame:
        text = content if isinstance(content, str) else _decode_taifex(content)
        raw = pd.read_csv(StringIO(text), dtype=str)
        raw.columns = [str(column).strip() for column in raw.columns]
        contract_col = _column(raw, "契約", "商品代號")
        date_col = _column(raw, "交易日期", "日期")
        expiry_col = _column(raw, "到期月份(週別)", "到期月份")
        tx = raw[raw[contract_col].astype(str).str.strip().eq("TX")].copy()
        if tx.empty:
            return pd.DataFrame(columns=YFINANCE_COLUMNS)

        tx["_volume"] = pd.to_numeric(tx[_column(tx, "成交量")].str.replace(",", ""), errors="coerce")
        tx = tx.sort_values("_volume").groupby(date_col, as_index=False).tail(1)
        created_at = pd.Timestamp.now(tz="UTC")
        frame = pd.DataFrame(
            {
                "datetime": pd.to_datetime(tx[date_col], format="%Y%m%d", errors="coerce"),
                "symbol": "TX",
                "market": "TW_DERIVATIVES",
                "timeframe": "1d",
                "open": _numeric_column(tx, "開盤價"),
                "high": _numeric_column(tx, "最高價"),
                "low": _numeric_column(tx, "最低價"),
                "close": _numeric_column(tx, "收盤價"),
                "volume": tx["_volume"],
                "source": self.source_name,
                "adjusted": False,
                "created_at": created_at,
            }
        )
        frame["contract"] = tx[expiry_col].to_numpy()
        return frame[YFINANCE_COLUMNS]

    def parse_put_call_ratio(self, frame: pd.DataFrame) -> pd.DataFrame:
        created_at = pd.Timestamp.now(tz="UTC")
        rows = []
        for _, item in frame.iterrows():
            rows.append(
                {
                    "datetime": item.get("日期", item.get("date")),
                    "symbol": "TXO",
                    "market": "TW_DERIVATIVES",
                    "dataset": "put_call_ratio",
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
                    "value": _number(
                        item.get("未平倉量買賣權比", item.get("put_call_open_interest_ratio"))
                    ),
                    "source": self.source_name,
                    "created_at": created_at,
                }
            )
        return pd.DataFrame(rows, columns=DERIVATIVE_COLUMNS)

    def parse_futures_derivatives_csv(self, content: bytes | str) -> pd.DataFrame:
        text = content if isinstance(content, str) else _decode_taifex(content)
        raw = pd.read_csv(StringIO(text), dtype=str)
        raw.columns = [str(column).strip() for column in raw.columns]
        contract_col = _column(raw, "契約", "商品代號")
        date_col = _column(raw, "交易日期", "日期")
        expiry_col = _column(raw, "到期月份(週別)", "到期月份")
        tx = raw[raw[contract_col].astype(str).str.strip().eq("TX")].copy()
        if tx.empty:
            return pd.DataFrame(columns=DERIVATIVE_COLUMNS)
        volume_col = _column(tx, "成交量")
        tx["_volume"] = pd.to_numeric(
            tx[volume_col].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
        tx = tx.sort_values("_volume").groupby(date_col, as_index=False).tail(1)
        created_at = pd.Timestamp.now(tz="UTC")
        rows = []
        for _, item in tx.iterrows():
            rows.append(
                {
                    "datetime": pd.to_datetime(
                        item.get(date_col),
                        format="%Y%m%d",
                        errors="coerce",
                    ),
                    "symbol": "TX",
                    "market": "TW_DERIVATIVES",
                    "dataset": "futures_open_interest",
                    "contract": "TX",
                    "expiry": item.get(expiry_col),
                    "option_type": None,
                    "strike": float("nan"),
                    "open": _number(item.get(_column(tx, "開盤價"))),
                    "high": _number(item.get(_column(tx, "最高價"))),
                    "low": _number(item.get(_column(tx, "最低價"))),
                    "close": _number(item.get(_column(tx, "收盤價"))),
                    "settlement": _number(
                        item.get(_optional_column(tx, "最後結算價", "結算價"))
                    ),
                    "volume": _number(item.get(volume_col)),
                    "open_interest": _number(
                        item.get(_optional_column(tx, "未沖銷契約數", "未平倉量"))
                    ),
                    "value": float("nan"),
                    "source": self.source_name,
                    "created_at": created_at,
                }
            )
        return pd.DataFrame(rows, columns=DERIVATIVE_COLUMNS)

    def parse_put_call_ratio_html(self, html: str) -> pd.DataFrame:
        parser = _TableParser()
        parser.feed(html)
        rows: list[dict] = []
        created_at = pd.Timestamp.now(tz="UTC")
        for cells in parser.rows:
            if len(cells) < 7 or not re.fullmatch(r"\d{4}/\d{1,2}/\d{1,2}", cells[0]):
                continue
            trading_day = pd.to_datetime(cells[0], errors="coerce")
            rows.extend(
                [
                    _derivative_metric_row(
                        trading_day,
                        "put_call_volume_ratio",
                        _number(cells[3]) / 100.0,
                        created_at,
                    ),
                    _derivative_metric_row(
                        trading_day,
                        "put_call_open_interest_ratio",
                        _number(cells[6]) / 100.0,
                        created_at,
                    ),
                ]
            )
        return pd.DataFrame(rows, columns=DERIVATIVE_COLUMNS)


def _decode_taifex(content: bytes) -> str:
    for encoding in ("utf-8-sig", "cp950", "big5"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("utf-8", errors="replace")


def _column(frame: pd.DataFrame, *candidates: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"TAIFEX data is missing columns: {candidates}")


def _optional_column(frame: pd.DataFrame, *candidates: str) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _numeric_column(frame: pd.DataFrame, name: str) -> pd.Series:
    return pd.to_numeric(frame[_column(frame, name)].str.replace(",", ""), errors="coerce")


def _number(value) -> float:
    cleaned = str(value).replace(",", "").replace("%", "") if value is not None else value
    converted = pd.to_numeric(pd.Series([cleaned]), errors="coerce").iloc[0]
    return float(converted) if pd.notna(converted) else float("nan")


def _derivative_metric_row(trading_day, dataset: str, value: float, created_at) -> dict:
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
        "source": "taifex",
        "created_at": created_at,
    }


class _TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append(" ".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None
