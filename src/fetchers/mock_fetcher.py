from __future__ import annotations

from datetime import date
import hashlib
import numpy as np
import pandas as pd

from .base_fetcher import BaseFetcher


class MockFetcher(BaseFetcher):
    """Synthetic data fetcher for offline testing.

    It creates deterministic OHLCV-like series with trend, cyclic behavior, noise,
    and occasional abnormal volatility. This lets the pipeline run even without
    internet access or API keys.
    """

    source_name = "mock"

    def fetch_price_history(
        self,
        symbols: list[str],
        start: str | date,
        end: str | date | None = None,
        interval: str = "1d",
    ) -> pd.DataFrame:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
        dates = pd.bdate_range(start_ts, end_ts)
        frames: list[pd.DataFrame] = []

        for symbol in symbols:
            seed = int(hashlib.md5(symbol.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            n = len(dates)
            base_price = 100 + (seed % 300)
            drift = rng.normal(0.0004, 0.0002)
            cycle = 0.01 * np.sin(np.arange(n) * 2 * np.pi / rng.integers(20, 80))
            shocks = rng.normal(drift, 0.018, n) + cycle

            # Inject deterministic anomaly near the end for demonstration.
            if n > 90:
                shocks[-10] += rng.choice([0.06, -0.06])
                shocks[-3] += rng.choice([0.04, -0.04])

            close = base_price * np.exp(np.cumsum(shocks))
            open_ = close * (1 + rng.normal(0, 0.004, n))
            high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0.006, 0.004, n)))
            low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0.006, 0.004, n)))
            volume = rng.integers(1_000_000, 40_000_000, n).astype(float)
            if n > 30:
                volume[-5] *= 2.8

            frame = pd.DataFrame(
                {
                    "datetime": dates,
                    "symbol": symbol,
                    "market": self._infer_market(symbol),
                    "timeframe": interval,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "source": self.source_name,
                    "adjusted": True,
                    "created_at": pd.Timestamp.now(tz="UTC"),
                }
            )
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    @staticmethod
    def _infer_market(symbol: str) -> str:
        if symbol.endswith(".TW") or symbol in {"TAIEX", "TX"}:
            return "TW"
        if symbol.endswith("-USD"):
            return "CRYPTO"
        return "US"
