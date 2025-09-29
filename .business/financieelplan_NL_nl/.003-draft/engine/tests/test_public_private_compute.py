from __future__ import annotations
from pathlib import Path
from pipelines.public import artifacts as pub
from pipelines.private import artifacts as prv
from core.loader import load_all
from .util_inputs import make_minimal_inputs


def test_public_private_compute_rows(tmp_path: Path):
    inputs = make_minimal_inputs(tmp_path)
    state = load_all(inputs)
    pub_tables = pub.compute_rows(state)
    prv_tables = prv.compute_rows(state)
    # Public outputs non-empty
    assert "public_vendor_choice" in pub_tables
    assert pub_tables["public_vendor_choice"][1], "vendor_choice rows empty"
    assert "public_tap_prices_per_model" in pub_tables
    assert pub_tables["public_tap_prices_per_model"][1]
    assert "public_tap_scenarios" in pub_tables
    assert pub_tables["public_tap_scenarios"][1]
    assert "public_tap_capacity_plan" in pub_tables
    assert pub_tables["public_tap_capacity_plan"][1]
    # Private outputs non-empty
    assert "private_tap_economics" in prv_tables
    assert prv_tables["private_tap_economics"][1]
    assert "private_tap_customers_by_month" in prv_tables
    assert prv_tables["private_tap_customers_by_month"][1]
