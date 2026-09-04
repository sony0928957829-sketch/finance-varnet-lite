"""Regression tests for the failures that kept the daily job red.

Root cause (2026-08-13 onwards): yfinance's `repair=True` path imports
scikit-learn, which yfinance does not declare as a dependency. requirements.txt
did not pin it either, so every download raised
ModuleNotFoundError("No module named 'sklearn'"), the health check saw an empty
market and the whole run aborted with exit code 1.

These tests lock in the three layers of the fix: the dependency itself, a
fetcher that degrades instead of returning nothing, and a health gate that
tolerates a partial outage without abandoning the day's report.
"""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.fetchers import yfinance_fetcher as yff
from src.fetchers.yfinance_fetcher import YFinanceFetcher
from src.health.data_health import evaluate_price_health, raise_for_health_errors

ROOT = Path(__file__).resolve().parents[1]


def test_requirements_pin_sklearn_for_yfinance_repair():
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "scikit-learn" in text, (
        "yfinance repair=True imports sklearn.cluster.DBSCAN but does not "
        "declare scikit-learn; it must be pinned here or every download fails."
    )
    assert "yfinance>=0.2,<2.0" in text, "yfinance needs an upper bound in CI."


def test_dependency_verification_runs_before_the_report():
    workflow = (ROOT / ".github/workflows/daily-market-report.yml").read_text(
        encoding="utf-8"
    )
    assert "Verify runtime dependencies" in workflow
    assert '"sklearn"' in workflow
    # A failed report must still commit the health evidence and file an issue.
    assert "steps.report.outputs.status" in workflow
    assert "market-report.log" in workflow


def _fake_frame(rows: int = 3) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=rows, freq="D", name="Date")
    return pd.DataFrame(
        {
            "Open": [10.0] * rows,
            "High": [11.0] * rows,
            "Low": [9.0] * rows,
            "Close": [10.5] * rows,
            "Volume": [1000.0] * rows,
        },
        index=index,
    )


class _FakeYF:
    """Stand-in for the yfinance module, recording how it was called."""

    def __init__(self, *, fail_on_repair=False, empty_attempts=0):
        self.fail_on_repair = fail_on_repair
        self.empty_attempts = empty_attempts
        self.calls = []

    def download(self, symbol, **kwargs):
        self.calls.append(kwargs.get("repair"))
        if self.fail_on_repair and kwargs.get("repair"):
            raise ModuleNotFoundError("No module named 'sklearn'")
        if self.empty_attempts > 0:
            self.empty_attempts -= 1
            return pd.DataFrame()
        return _fake_frame()


def test_download_falls_back_to_unrepaired_bars(monkeypatch):
    """A broken repair path must cost quality, not the entire symbol."""
    monkeypatch.setattr(yff, "sklearn_available", lambda: True)
    monkeypatch.setattr(yff, "RETRY_BACKOFF_SECONDS", 0)
    fake = _FakeYF(fail_on_repair=True)

    data = YFinanceFetcher._download(
        fake, "NVDA", start="2026-01-01", end="2026-01-05", interval="1d"
    )

    assert data is not None and not data.empty
    assert True in fake.calls and False in fake.calls


def test_download_retries_transient_empty_responses(monkeypatch):
    monkeypatch.setattr(yff, "sklearn_available", lambda: True)
    monkeypatch.setattr(yff, "RETRY_BACKOFF_SECONDS", 0)
    fake = _FakeYF(empty_attempts=2)

    data = YFinanceFetcher._download(
        fake, "NVDA", start="2026-01-01", end="2026-01-05", interval="1d"
    )

    assert data is not None and not data.empty
    assert len(fake.calls) == 3


def test_download_skips_repair_when_sklearn_is_absent(monkeypatch):
    monkeypatch.setattr(yff, "sklearn_available", lambda: False)
    monkeypatch.setattr(yff, "RETRY_BACKOFF_SECONDS", 0)
    fake = _FakeYF()

    YFinanceFetcher._download(
        fake, "NVDA", start="2026-01-01", end="2026-01-05", interval="1d"
    )

    assert fake.calls == [False]


HEALTH_CONFIG = {
    "required_columns": [
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
    ],
    "minimum_rows_per_symbol": 2,
    "freshness_max_age_days": {"default": 4},
    "long_gap_days": {"default": 4},
    "fail_on": [
        "empty_data",
        "missing_columns",
        "missing_symbol",
        "insufficient_rows",
        "invalid_datetime",
        "null_price",
        "non_positive_price",
        "invalid_ohlc",
        "duplicate_bar",
        "stale_data",
    ],
    "degraded_mode": {
        "enabled": True,
        "minimum_symbol_coverage": 0.6,
        "tolerated_codes": [
            "missing_symbol",
            "insufficient_rows",
            "stale_data",
            "long_gap",
        ],
        "critical_symbols": ["2330.TW", "TAIEX"],
    },
}


def _price_frame(symbols, as_of=date(2026, 1, 10), rows=5, bad_ohlc=False):
    frames = []
    for symbol in symbols:
        index = pd.date_range(end=pd.Timestamp(as_of), periods=rows, freq="D")
        frame = pd.DataFrame(
            {
                "datetime": index,
                "symbol": symbol,
                "market": "TW" if symbol.endswith(".TW") or symbol == "TAIEX" else "US",
                "timeframe": "1d",
                "open": 10.0,
                "high": 5.0 if bad_ohlc else 11.0,
                "low": 9.0,
                "close": 10.5,
                "volume": 1000.0,
                "source": "yfinance",
            }
        )
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


EXPECTED = ["2330.TW", "TAIEX", "2317.TW", "NVDA", "TSLA"]


def test_partial_outage_degrades_instead_of_aborting():
    frame = _price_frame(["2330.TW", "TAIEX", "2317.TW", "NVDA"])

    report = evaluate_price_health(
        frame,
        expected_symbols=EXPECTED,
        as_of=date(2026, 1, 10),
        primary_source="yfinance",
        config=HEALTH_CONFIG,
    )

    assert report["degraded"]["active"] is True
    assert report["degraded"]["missing_symbols"] == ["TSLA"]
    assert report["status"] == "warning"
    raise_for_health_errors(report)  # must not raise


def test_missing_critical_symbol_still_fails():
    frame = _price_frame(["TAIEX", "2317.TW", "NVDA", "TSLA"])

    report = evaluate_price_health(
        frame,
        expected_symbols=EXPECTED,
        as_of=date(2026, 1, 10),
        primary_source="yfinance",
        config=HEALTH_CONFIG,
    )

    assert report["degraded"]["active"] is False
    assert report["status"] == "error"
    with pytest.raises(Exception):
        raise_for_health_errors(report)


def test_coverage_below_minimum_still_fails():
    frame = _price_frame(["2330.TW", "TAIEX"])

    report = evaluate_price_health(
        frame,
        expected_symbols=EXPECTED,
        as_of=date(2026, 1, 10),
        primary_source="yfinance",
        config=HEALTH_CONFIG,
    )

    assert report["degraded"]["active"] is False
    assert report["status"] == "error"


def test_corrupt_prices_are_never_tolerated():
    frame = _price_frame(EXPECTED, bad_ohlc=True)

    report = evaluate_price_health(
        frame,
        expected_symbols=EXPECTED,
        as_of=date(2026, 1, 10),
        primary_source="yfinance",
        config=HEALTH_CONFIG,
    )

    assert report["degraded"]["active"] is False
    assert report["status"] == "error"


def test_empty_market_still_fails():
    report = evaluate_price_health(
        pd.DataFrame(),
        expected_symbols=EXPECTED,
        as_of=date(2026, 1, 10),
        primary_source="yfinance",
        config=HEALTH_CONFIG,
    )

    assert report["degraded"]["active"] is False
    assert report["status"] == "error"


def test_healthy_run_is_not_labelled_degraded():
    """A clean run must not report itself as degraded (run #181 said 13/13)."""
    frame = _price_frame(EXPECTED)

    report = evaluate_price_health(
        frame,
        expected_symbols=EXPECTED,
        as_of=date(2026, 1, 10),
        primary_source="yfinance",
        config=HEALTH_CONFIG,
    )

    assert report["degraded"]["active"] is False
    assert report["degraded"]["missing_symbols"] == []
    assert report["status"] in {"healthy", "warning"}
