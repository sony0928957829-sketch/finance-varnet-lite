from __future__ import annotations

import numpy as np
import pandas as pd

from src.report.validation_summary import build_validation_summary


def _history():
    rows = []
    for i, day in enumerate(pd.date_range("2026-06-01", periods=8, freq="D")):
        rows.append(dict(as_of=day, signal="momentum_score", horizon=5, n=300,
                         direction_hit_rate=0.56, direction_n=300, hit_zstat=2.4,
                         ic=0.10 + i * 0.01, ic_tstat=3.5))
        rows.append(dict(as_of=day, signal="noise_x", horizon=5, n=300,
                         direction_hit_rate=0.49, direction_n=300, hit_zstat=-0.3,
                         ic=-0.01, ic_tstat=-0.4))
        rows.append(dict(as_of=day, signal="thin", horizon=5, n=5,
                         direction_hit_rate=np.nan, direction_n=0, hit_zstat=np.nan,
                         ic=np.nan, ic_tstat=np.nan))
    return pd.DataFrame(rows)


def test_summary_flags_edge_noise_and_insufficient():
    text = build_validation_summary(_history())
    assert "顯著正向 edge" in text
    assert "momentum_score" in text
    assert "資料不足" in text          # thin signal
    assert "無顯著 edge" in text        # noise


def test_empty_history():
    text = build_validation_summary(pd.DataFrame())
    assert "還沒" in text or "空" in text
