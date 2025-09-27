from __future__ import annotations

import csv

from finance_engine.config import OUTPUTS
from finance_engine.engine_pkg import run as engine_run


def _read_csv_head(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return reader.fieldnames or [], rows


def test_model_price_csv_schema():
    assert engine_run() == 0
    cols, rows = _read_csv_head(OUTPUTS / "model_price_per_1m_tokens.csv")
    for k in [
        "model",
        "gpu",
        "tps",
        "cost_per_1m_min",
        "cost_per_1m_med",
        "cost_per_1m_max",
        "sell_per_1m_eur",
        "margin_per_1m_min",
        "margin_per_1m_med",
        "margin_per_1m_max",
        "gross_margin_pct_med",
    ]:
        assert k in cols, f"Missing column in model_price_per_1m_tokens.csv: {k}"
    assert len(rows) > 0


def test_public_scenarios_csv_schema():
    assert engine_run() == 0
    cols, rows = _read_csv_head(OUTPUTS / "public_tap_scenarios.csv")
    for k in [
        "case",
        "m_tokens",
        "revenue_eur",
        "cogs_eur",
        "gross_margin_eur",
        "gross_margin_pct",
        "marketing_reserved_eur",
        "fixed_plus_loan_eur",
        "net_eur",
    ]:
        assert k in cols, f"Missing column in public_tap_scenarios.csv: {k}"
    assert len(rows) >= 3
