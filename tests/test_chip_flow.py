from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.chip_flow import _institutional_flow_score


def _mixed_unit_chip(seed: int = 0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=90, freq="B")
    rows = []
    for sym in ["2330.TW", "2317.TW"]:
        for d in dates:
            rows.append({"symbol": sym, "datetime": d, "metric": "foreign_net_buy", "value": rng.normal(0, 5_000)})
            rows.append({"symbol": sym, "datetime": d, "metric": "dealer_net_buy_value", "value": rng.normal(0, 3e8)})
    return pd.DataFrame(rows)


def test_mixed_units_no_longer_skipped():
    out = _institutional_flow_score(_mixed_unit_chip())
    assert out["institutional_flow_score"].notna().sum() > 0
    vals = out["institutional_flow_score"].dropna()
    assert vals.between(0, 100).all()
    assert 30 < vals.mean() < 70  # roughly centred, not degenerate


def test_empty_chip_degrades():
    out = _institutional_flow_score(pd.DataFrame())
    assert out.empty
    assert list(out.columns) == ["symbol", "datetime", "institutional_flow_score"]


def test_no_institutional_rows_degrades():
    df = pd.DataFrame({"symbol": ["A"], "datetime": pd.to_datetime(["2025-01-01"]),
                       "metric": ["unrelated_pe_ratio"], "value": [10.0]})
    out = _institutional_flow_score(df)
    assert out.empty
