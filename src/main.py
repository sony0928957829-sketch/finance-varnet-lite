from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from zoneinfo import ZoneInfo
from pathlib import Path

import pandas as pd

from src.fetchers.factory import create_fetcher
from src.fetchers.router import fetch_prices_with_fallback
from src.health.data_health import (
    evaluate_price_health,
    raise_for_health_errors,
    write_health_report,
)
from src.normalizers.price_normalizer import normalize_price_frame
from src.features.basic import add_basic_features
from src.features.fourier import add_fourier_features
from src.features.wavelet import add_wavelet_features
from src.features.relative_strength import add_relative_strength
from src.features.macro_context import add_macro_context
from src.features.labels import add_future_range_labels
from src.models.range_forecast import add_range_forecasts
from src.pipeline.supplemental import collect_supplemental_data
from src.storage.lake import archive_pipeline_run
from src.features.chip_flow import add_chip_flow_features
from src.features.factor_standardize import add_standardized_factors, standardized_columns
from src.models.ml_forecast import add_ml_forecast, ml_forecast_column
from src.evaluation.signal_validation import evaluate_signals, update_track_record
from src.evaluation.backtest import run_backtest, write_backtest_report
from src.report.validation_summary import write_validation_summary
from src.scoring.scores import add_scores
from src.report.daily_report import generate_daily_report
from src.utils.config import DATA_DIR, PROJECT_ROOT, ensure_dirs, load_config
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


def _iter_price_compatible_routes(config: dict):
    datasets = config.get("datasets", {})
    yield from datasets.get("prices", {}).values()
    yield from datasets.get("macro", {}).values()


def provider_price_settings(config: dict, provider: str) -> tuple[list[str], int]:
    selected_symbols: list[str] = []
    history_years = 0

    for route in _iter_price_compatible_routes(config):
        if not route.get("enabled", False) or route.get("primary") != provider:
            continue
        selected_symbols.extend(route.get("symbols", []))
        history_years = max(history_years, int(route.get("history_years", 0)))

    return list(dict.fromkeys(selected_symbols)), history_years


def provider_symbol_aliases(config: dict, provider: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for route in _iter_price_compatible_routes(config):
        if not route.get("enabled", False) or route.get("primary") != provider:
            continue
        aliases.update(route.get("provider_symbols", {}).get(provider, {}))
    return aliases


ML_FACTOR_COLUMNS = [
    "return_5d",
    "return_20d",
    "return_60d",
    "volume_ratio",
    "atr_pct",
    "relative_strength_20d",
    "fourier_cycle_strength",
    "wavelet_anomaly_score",
    "institutional_flow_score",
    "options_sentiment_score",
    "condition_score",
]
ML_HORIZON = 5


def _validate_signals(features, *, mode: str, report_date) -> None:
    """Score how well each signal predicts 5-day forward returns and persist it.

    Writes a dated scorecard snapshot and appends to a per-mode track record.
    Never raises into the main pipeline: validation is observability, not a
    gate, so any failure is swallowed after the data products are already saved.
    """
    try:
        candidate_signals = [
            "trend_score",
            "momentum_score",
            "relative_strength_score",
            "condition_score",
            "institutional_flow_score",
            "options_sentiment_score",
        ]
        signals = [c for c in candidate_signals if c in features.columns]
        scorecards = []
        if signals:
            scorecards.append(evaluate_signals(features, signals, horizons=(5,)))
        ml_col = ml_forecast_column(ML_HORIZON)
        if ml_col in features.columns:
            # The ML forecast is a predicted return centred at 0, so its
            # "direction" pivot is 0 (not 50 like the 0-100 scores).
            scorecards.append(
                evaluate_signals(features, [ml_col], horizons=(ML_HORIZON,), neutral=0.0)
            )
        if not scorecards:
            return
        scorecard = pd.concat(scorecards, ignore_index=True)

        eval_dir = DATA_DIR / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)
        scorecard.to_csv(
            eval_dir / f"{report_date.isoformat()}_signal_scorecard_{mode}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        update_track_record(
            scorecard,
            eval_dir / f"signal_track_record_{mode}.parquet",
            as_of=pd.Timestamp(report_date),
        )
    except Exception as exc:  # pragma: no cover - observability must not break runs
        print(f"[warn] signal validation skipped: {exc}")


def run_pipeline(mode: str = "mock", start: str | None = None, end: str | None = None) -> Path:
    ensure_dirs()
    watchlist_config = load_config("watchlist.yaml")
    feature_config = load_config("feature_config.yaml")
    data_sources_config = load_config("data_sources.yaml")
    data_health_config = load_config("data_health.yaml")
    storage_config = load_config("storage.yaml")
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

    price_route_status = {}
    if mode == "yfinance":
        raw, price_route_status = fetch_prices_with_fallback(
            data_sources_config,
            primary_provider=mode,
            start=start,
            end=end,
            interval="1d",
        )
        optional_prices, optional_route_status = fetch_prices_with_fallback(
            data_sources_config,
            primary_provider="taifex",
            start=start,
            end=end,
            interval="1d",
        )
        price_route_status.update(optional_route_status)
        if not optional_prices.empty:
            raw = pd.concat([raw, optional_prices], ignore_index=True).drop_duplicates(
                subset=["datetime", "symbol", "timeframe"],
                keep="first",
            )
            all_symbols.extend(
                symbol
                for symbol in optional_prices["symbol"].unique()
                if symbol not in all_symbols
            )
    else:
        fetcher = create_fetcher(mode)
        raw = fetcher.fetch_price_history(all_symbols, start=start, end=end, interval="1d")
    report_date = current_taipei_date()
    health_report = evaluate_price_health(
        raw,
        expected_symbols=all_symbols,
        as_of=date.fromisoformat(end),
        primary_source=mode,
        config=data_health_config,
    )
    health_path = DATA_DIR / "reports" / f"{report_date.isoformat()}_data_health.json"
    write_health_report(health_report, health_path)
    raise_for_health_errors(health_report)

    normalized = normalize_price_frame(raw)

    # Save normalized data for inspection.
    normalized_path = DATA_DIR / "normalized" / f"prices_{mode}.parquet"
    save_frame(normalized, normalized_path, parquet_required=mode == "yfinance")

    # Collect chip/options supplemental data BEFORE feature engineering so the
    # chip_flow features can read today's freshly downloaded files (not yesterday's).
    supplemental_status = {}
    if mode == "yfinance":
        supplemental_status["price_routes"] = {
            "status": (
                "ok"
                if all(item["status"] == "ok" for item in price_route_status.values())
                else "warning"
            ),
            "rows": int(len(raw)),
            "routes": price_route_status,
        }
        supplemental_status = collect_supplemental_data(
            data_sources_config,
            symbols=all_symbols,
            start=start,
            end=end,
            output_dir=DATA_DIR / "alternative",
        ) | supplemental_status
        supplemental_path = (
            DATA_DIR / "reports" / f"{report_date.isoformat()}_supplemental_health.json"
        )
        supplemental_path.write_text(
            json.dumps(supplemental_status, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

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
    features = add_macro_context(features)
    forecast_config = feature_config.get("range_forecast", {})
    features = add_range_forecasts(
        features,
        horizons=feature_config.get("prediction_targets_reserved", {}).get(
            "horizons",
            [1, 5, 10],
        ),
        window=int(forecast_config.get("window", 252)),
        min_periods=int(forecast_config.get("min_periods", 60)),
        lower_quantile=float(forecast_config.get("lower_quantile", 0.20)),
        upper_quantile=float(forecast_config.get("upper_quantile", 0.80)),
    )
    features = add_chip_flow_features(features, alt_dir=DATA_DIR / "alternative")
    features = add_scores(features)

    # Learned layer: standardize factors cross-sectionally (per market/day), then
    # let a walk-forward model learn how to combine them into a 5-day return
    # forecast. This replaces hand-picked weights with data-learned ones; the
    # forecast is measured by the validation layer, not treated as advice.
    features = add_standardized_factors(
        features, ML_FACTOR_COLUMNS, group_columns=("datetime", "market")
    )
    features = add_ml_forecast(
        features,
        standardized_columns(ML_FACTOR_COLUMNS),
        horizon=ML_HORIZON,
        model="auto",  # uses LightGBM if installed, else numpy ridge
    )

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

    # Self-validation: does each 0-100 signal actually predict 5-day forward
    # returns? Append today's scorecard to a track record so the edge (or lack
    # of it) accumulates over time. This measures the signals; it is not advice.
    _validate_signals(features, mode=mode, report_date=report_date)

    # Readable weekly summary of the accumulating validation track record.
    try:
        write_validation_summary(
            DATA_DIR / "evaluation" / f"signal_track_record_{mode}.parquet",
            DATA_DIR / "reports" / f"{report_date.isoformat()}_validation_summary_{mode}.md",
            as_of=report_date,
        )
    except Exception as exc:  # pragma: no cover - reporting must not break runs
        print(f"[warn] validation summary skipped: {exc}")

    # Cost-aware backtest: would acting on each signal have beaten buy-and-hold?
    try:
        backtest_signals = [
            ml_forecast_column(ML_HORIZON),
            "condition_score",
            "momentum_score",
        ]
        results = [
            run_backtest(features, col, horizon=ML_HORIZON, top_k=3, cost_bps=10.0)
            for col in backtest_signals
            if col in features.columns
        ]
        if results:
            write_backtest_report(
                results,
                DATA_DIR / "reports" / f"{report_date.isoformat()}_backtest_{mode}.md",
                cost_bps=10.0,
                top_k=3,
            )
    except Exception as exc:  # pragma: no cover - reporting must not break runs
        print(f"[warn] backtest skipped: {exc}")

    report_path = DATA_DIR / "reports" / f"{report_date.isoformat()}_market_report.md"
    generate_daily_report(
        features,
        report_path,
        title_date=report_date,
        health_report=health_report,
        supplemental_status=supplemental_status,
    )

    # Archive this run into the local partitioned data lake. The Drive sync is a
    # separate workflow step (src.storage.cli push) so mock stays offline-safe.
    lake_config = storage_config.get("data_lake", {})
    if mode in lake_config.get("archive_modes", ["yfinance"]):
        lake_root = Path(lake_config.get("local_root", "data/lake"))
        if not lake_root.is_absolute():
            lake_root = PROJECT_ROOT / lake_root
        archive_pipeline_run(
            lake_root=lake_root,
            raw=raw,
            normalized=normalized,
            features=features,
            labels=labeled_features,
            alternative_dir=DATA_DIR / "alternative",
            evaluation_dir=DATA_DIR / "evaluation",
            report_date=report_date,
            mode=mode,
            model_version=lake_config.get("model_version", "range-forecast-v1"),
        )
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
