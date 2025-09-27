from __future__ import annotations

import pandas as pd

from finance_engine.compute.vat import vat_examples
from finance_engine.validation.validate import validate_inputs


def test_vat_examples_21pct():
    rows = vat_examples(21.0)
    assert rows == [
        {"inflow_eur": 1000, "vat_set_aside_eur": 210.0},
        {"inflow_eur": 10000, "vat_set_aside_eur": 2100.0},
        {"inflow_eur": 100000, "vat_set_aside_eur": 21000.0},
    ]


def test_validate_inputs_missing_columns():
    config = {
        "currency": "EUR",
        "pricing_inputs": {"fx_buffer_pct": 5},
        "tax_billing": {"vat_standard_rate_pct": 21.0},
        "finance": {"marketing_allocation_pct_of_inflow": 15},
    }
    costs = {}
    lending = {}
    price_sheet = pd.DataFrame({"sku": ["A"]})  # incomplete columns
    report = validate_inputs(config=config, costs=costs, lending=lending, price_sheet=price_sheet)
    # Expect errors for missing required columns in price_sheet
    assert any("missing column" in e for e in report["errors"])  # at least one error
