from datetime import date
from pathlib import Path

import pandas as pd

from src.storage.lake import archive_frame, build_prediction_snapshot


def _price_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2026-05-29", "2026-06-01", "2026-06-02"],
                utc=True,
            ),
            "symbol": ["2330.TW"] * 3,
            "market": ["TW"] * 3,
            "timeframe": ["1d"] * 3,
            "close": [1000.0, 1010.0, 1020.0],
            "pred_next_5d_high_pct": [0.02, 0.03, 0.04],
            "next_5d_high_pct": [0.01, 0.02, float("nan")],
        }
    )


def test_archive_frame_partitions_by_month_and_is_idempotent(tmp_path: Path):
    frame = _price_frame()
    first = archive_frame(
        frame,
        lake_root=tmp_path,
        layer="normalized",
        dataset="prices_yfinance",
    )
    second = archive_frame(
        frame,
        lake_root=tmp_path,
        layer="normalized",
        dataset="prices_yfinance",
    )

    assert len(first) == 2
    assert first == second
    assert sum(len(pd.read_parquet(path)) for path in first) == len(frame)


def test_prediction_snapshot_excludes_future_labels():
    snapshot = build_prediction_snapshot(
        _price_frame(),
        prediction_date=date(2026, 6, 2),
        model_version="range-v1",
        data_version="abc123",
    )

    assert len(snapshot) == 1
    assert "pred_next_5d_high_pct" in snapshot.columns
    assert "next_5d_high_pct" not in snapshot.columns
    assert snapshot.iloc[0]["model_version"] == "range-v1"
    assert snapshot.iloc[0]["data_version"] == "abc123"
    assert snapshot.iloc[0]["input_cutoff"] == pd.Timestamp(
        "2026-06-02",
        tz="UTC",
    )
