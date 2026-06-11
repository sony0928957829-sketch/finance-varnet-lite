from __future__ import annotations

import argparse
from datetime import date, datetime
from zoneinfo import ZoneInfo
from pathlib import Path

from src.fetchers.factory import create_fetcher
from src.normalizers.price_normalizer import normalize_price_frame
from src.features.basic import add_basic_features
from src.features.fourier import add_fourier_features
from src.features.wavelet import add_wavelet_features
from src.features.relative_strength import add_relative_strength
from src.features.labels import add_future_range_labels
from src.scoring.scores import add_scores
from src.report.daily_report import generate_daily_report
from src.utils.config import DATA_DIR, ensure_dirs, load_config
from src.utils.watchlist import flatten_watchlist, symbols


def current_taipei_date() -> date:
    return datetime.now(ZoneInfo("Asia/Taipei")).date()


def years_before(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year - years)
    except ValueError:
        return day.replace(year=day.year - years, month=2, day=28)


def save_frame(frame, path: Path, *, parquet_required: bool = False) -> Path:
    try:
        frame.to_parquet(path, index=False)
        return path
    except Exception as exc:
        if parquet_required:
            raise RuntimeError(
                f"Failed to write required Parquet output: {path}. "
                "Install the project requirements, including pyarrow."
            ) from exc
        csv_path = path.with_suffix(".csv")
        frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return csv_path


def provider_price_settings(config: dict, provider: str) -> tuple[list[str], int]:
    selected_symbols: list[str] = []
    history_years = 0
    price_routes = config.get("datasets", {}).get("prices", {})

    for route in price_routes.values():
        if not route.get("enabled", False) or route.get("primary") != provider:
            continue
        selected_symbols.extend(route.get("symbols", []))
        history_years = max(history_years, int(route.get("history_years", 0)))

    return list(dict.fromkeys(selected_symbols)), history_years


def run_pipeline(mode: str = "mock", start: str | None = None, end: str | None = None) -> Path:
    ensure_dirs()
    watchlist_config = load_config("watchlist.yaml")
    feature_config = load_config("feature_config.yaml")
    data_sources_config = load_config("data_sources.yaml")
    instruments = flatten_watchlist(watchlist_config)
    all_symbols = symbols(instruments)

    if mode == "yfinance":
        all_symbols, configured_history_years = provider_price_settings(
            data_sources_config,
            provider="yfinance",
        )
        if not all_symbols:
            raise ValueError("No enabled yfinance price routes are configured.")
    else:
        configured_history_years = 3

    if start is None:
        start = years_before(
            current_taipei_date(),
            configured_history_years or 5,
        ).isoformat()
    if end is None:
        end = current_taipei_date().isoformat()

    fetcher = create_fetcher(mode)
    raw = fetcher.fetch_price_history(all_symbols, start=start, end=end, interval="1d")
    normalized = normalize_price_frame(raw)

    # Save normalized data for inspection.
    normalized_path = DATA_DIR / "normalized" / f"prices_{mode}.parquet"
    save_frame(normalized, normalized_path, parquet_required=mode == "yfinance")

    features = add_basic_features(normalized)
    features = add_fourier_features(
        features,
        window=feature_config.get("fourier", {}).get("window", 120),
        min_period=feature_config.get("fourier", {}).get("min_period", 5),
        max_period=feature_config.get("fourier", {}).get("max_period", 80),
    )
    features = add_wavelet_features(
        features,
        window=feature_config.get("wavelet", {}).get("window", 120),
        wavelet_name=feature_config.get("wavelet", {}).get("wavelet_name", "db4"),
        level=feature_config.get("wavelet", {}).get("level", 3),
    )

    benchmark_map = {instrument.symbol: instrument.benchmark for instrument in instruments}
    features = add_relative_strength(features, benchmark_map)
    features = add_scores(features)

    features_path = DATA_DIR / "features" / f"features_{mode}.parquet"
    save_frame(features, features_path, parquet_required=mode == "yfinance")

    # Training/backtest labels are stored separately from live inference features.
    labeled_features = add_future_range_labels(
        features,
        horizons=feature_config.get("prediction_targets_reserved", {}).get(
            "horizons",
            [1, 5, 10],
        ),
    )
    labels_path = DATA_DIR / "labels" / f"labels_{mode}.parquet"
    save_frame(labeled_features, labels_path, parquet_required=mode == "yfinance")

    report_date = current_taipei_date()
    report_path = DATA_DIR / "reports" / f"{report_date.isoformat()}_market_report.md"
    generate_daily_report(features, report_path, title_date=report_date)
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VARnet-lite v0.1 market intelligence pipeline.")
    parser.add_argument("--mode", choices=["mock", "yfinance"], default="mock", help="Data mode to use.")
    parser.add_argument("--start", default=None, help="Start date, e.g. 2021-01-01")
    parser.add_argument("--end", default=None, help="End date, e.g. 2026-06-11")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = run_pipeline(mode=args.mode, start=args.start, end=args.end)
    print(f"Report generated: {output}")
