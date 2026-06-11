from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from src.normalizers.alternative_data import NEWS_COLUMNS


class YFinanceNewsFetcher:
    source_name = "yahoo_finance_news"

    def fetch_news(
        self,
        symbols: list[str],
        start: str | date | None = None,
        end: str | date | None = None,
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise RuntimeError("yfinance is required for Yahoo Finance news.") from exc

        rows: list[dict] = []
        created_at = pd.Timestamp.now(tz="UTC")
        for symbol in symbols:
            for item in yf.Ticker(symbol).news or []:
                content = item.get("content", item)
                timestamp = content.get("pubDate") or content.get("providerPublishTime")
                rows.append(
                    {
                        "datetime": _news_datetime(timestamp),
                        "symbol": symbol,
                        "market": _market(symbol),
                        "event_type": content.get("contentType", "news"),
                        "title": content.get("title", ""),
                        "summary": content.get("summary", content.get("description", "")),
                        "url": _news_url(content),
                        "publisher": _publisher(content),
                        "source": self.source_name,
                        "created_at": created_at,
                    }
                )
        frame = pd.DataFrame(rows, columns=NEWS_COLUMNS)
        if frame.empty:
            return frame
        timestamps = pd.to_datetime(frame["datetime"], errors="coerce", utc=True)
        if start is not None:
            timestamps_start = pd.Timestamp(start, tz="UTC")
            frame = frame[timestamps.ge(timestamps_start)]
            timestamps = timestamps.loc[frame.index]
        if end is not None:
            timestamps_end = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
            frame = frame[timestamps.lt(timestamps_end)]
        return frame.reset_index(drop=True)


def _news_datetime(value):
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    return pd.to_datetime(value, errors="coerce", utc=True)


def _news_url(content: dict) -> str:
    canonical = content.get("canonicalUrl") or {}
    click_through = content.get("clickThroughUrl") or {}
    return canonical.get("url") or click_through.get("url") or content.get("link", "")


def _publisher(content: dict) -> str:
    provider = content.get("provider") or {}
    return provider.get("displayName") or content.get("publisher", "")


def _market(symbol: str) -> str:
    if symbol.endswith(".TW") or symbol == "TAIEX":
        return "TW"
    if symbol.endswith("-USD"):
        return "CRYPTO"
    return "US"
