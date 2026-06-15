from __future__ import annotations

from datetime import date
import hashlib
from pathlib import Path
import re
import shutil

import pandas as pd


FUTURE_LABEL_PATTERN = re.compile(r"^next_\d+d_(high|low)_pct$")


def _safe_segment(value) -> str:
    text = str(value if pd.notna(value) else "unknown").strip() or "unknown"
    return re.sub(r"[^A-Za-z0-9._=-]+", "_", text)


def _frame_version(frame: pd.DataFrame) -> str:
    columns = [
        column
        for column in ("datetime", "symbol", "close", "source")
        if column in frame.columns
    ]
    if not columns or frame.empty:
        return hashlib.sha256(b"empty").hexdigest()[:16]
    stable = frame[columns].copy().sort_values(columns).reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(stable, index=False).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()[:16]


def _merge_partition(
    frame: pd.DataFrame,
    path: Path,
    *,
    dedupe_keys: tuple[str, ...],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    combined = frame.copy()
    if path.exists():
        combined = pd.concat(
            [pd.read_parquet(path), combined],
            ignore_index=True,
        )
    keys = [key for key in dedupe_keys if key in combined.columns]
    if keys:
        combined = combined.drop_duplicates(subset=keys, keep="last")
        combined = combined.sort_values(keys).reset_index(drop=True)
    else:
        combined = combined.drop_duplicates().reset_index(drop=True)
    combined.to_parquet(path, index=False)
    return path


def archive_frame(
    frame: pd.DataFrame,
    *,
    lake_root: Path,
    layer: str,
    dataset: str,
    dedupe_keys: tuple[str, ...] = ("datetime", "symbol", "timeframe"),
) -> list[Path]:
    if frame.empty or "datetime" not in frame.columns:
        return []
    work = frame.copy()
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce", utc=True)
    work = work.dropna(subset=["datetime"])
    if work.empty:
        return []
    for column, default in (
        ("market", "unknown"),
        ("symbol", "all"),
        ("timeframe", "na"),
    ):
        if column not in work.columns:
            work[column] = default
    work["_year"] = work["datetime"].dt.year
    work["_month"] = work["datetime"].dt.month

    paths = []
    group_columns = ["market", "symbol", "timeframe", "_year", "_month"]
    for keys, partition in work.groupby(group_columns, dropna=False, sort=True):
        market, symbol, timeframe, year, month = keys
        path = (
            lake_root
            / layer
            / _safe_segment(dataset)
            / f"market={_safe_segment(market)}"
            / f"symbol={_safe_segment(symbol)}"
            / f"timeframe={_safe_segment(timeframe)}"
            / f"year={int(year):04d}"
            / f"month={int(month):02d}"
            / "part.parquet"
        )
        paths.append(
            _merge_partition(
                partition.drop(columns=["_year", "_month"]),
                path,
                dedupe_keys=dedupe_keys,
            )
        )
    return paths


def build_prediction_snapshot(
    features: pd.DataFrame,
    *,
    prediction_date: date,
    model_version: str,
    data_version: str,
) -> pd.DataFrame:
    if features.empty:
        return features.copy()
    label_columns = {
        column for column in features.columns if FUTURE_LABEL_PATTERN.match(column)
    }
    allowed = [
        column
        for column in features.columns
        if column not in label_columns
        and (
            column
            in {
                "datetime",
                "symbol",
                "market",
                "timeframe",
                "source",
                "close",
                "risk_score",
                "trend_score",
                "momentum_score",
                "relative_strength_score",
                "condition_score",
            }
            or column.startswith("pred_next_")
            or column.endswith("_anomaly")
        )
    ]
    latest = (
        features.sort_values(["symbol", "datetime"])
        .groupby("symbol", as_index=False)
        .tail(1)[allowed]
        .copy()
    )
    latest["prediction_as_of"] = pd.Timestamp(prediction_date, tz="UTC")
    latest["input_cutoff"] = pd.to_datetime(
        latest["datetime"],
        errors="coerce",
        utc=True,
    )
    latest["model_version"] = model_version
    latest["data_version"] = data_version
    return latest


def _archive_existing_files(
    source_dir: Path,
    destination: Path,
    patterns: tuple[str, ...],
) -> list[Path]:
    paths = []
    if not source_dir.exists():
        return paths
    for pattern in patterns:
        for source in sorted(source_dir.glob(pattern)):
            if not source.is_file():
                continue
            target = destination / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            paths.append(target)
    return paths


def archive_pipeline_run(
    *,
    lake_root: Path,
    raw: pd.DataFrame,
    normalized: pd.DataFrame,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    alternative_dir: Path,
    evaluation_dir: Path,
    report_date: date,
    mode: str,
    model_version: str,
) -> list[Path]:
    """Persist one run into mergeable Parquet partitions and snapshots."""
    paths = []
    paths.extend(
        archive_frame(
            raw,
            lake_root=lake_root,
            layer="raw",
            dataset=f"prices_{mode}",
        )
    )
    paths.extend(
        archive_frame(
            normalized,
            lake_root=lake_root,
            layer="normalized",
            dataset=f"prices_{mode}",
        )
    )
    paths.extend(
        archive_frame(
            features,
            lake_root=lake_root,
            layer="features",
            dataset=f"features_{mode}",
        )
    )
    paths.extend(
        archive_frame(
            labels,
            lake_root=lake_root,
            layer="labels",
            dataset=f"labels_{mode}",
        )
    )

    predictions = build_prediction_snapshot(
        features,
        prediction_date=report_date,
        model_version=model_version,
        data_version=_frame_version(normalized),
    )
    paths.extend(
        archive_frame(
            predictions,
            lake_root=lake_root,
            layer="predictions",
            dataset=f"range_forecast_{mode}",
            dedupe_keys=("prediction_as_of", "symbol", "model_version"),
        )
    )

    for source in sorted(alternative_dir.glob("*.parquet")):
        try:
            frame = pd.read_parquet(source)
        except Exception:
            continue
        paths.extend(
            archive_frame(
                frame,
                lake_root=lake_root,
                layer="alternative",
                dataset=source.stem,
                dedupe_keys=(
                    "datetime",
                    "symbol",
                    "dataset",
                    "metric",
                    "contract",
                    "expiry",
                    "strike",
                    "url",
                ),
            )
        )

    paths.extend(
        _archive_existing_files(
            evaluation_dir,
            lake_root / "evaluation" / mode,
            ("*.csv", "*.parquet"),
        )
    )
    return paths
