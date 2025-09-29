from __future__ import annotations
from pathlib import Path
import shutil
from d3_engine.runner import runner
from .util_inputs import make_minimal_inputs


def test_runner_smoke(tmp_path: Path):
    inputs = make_minimal_inputs(tmp_path)
    out1 = tmp_path / "out1"
    res = runner.execute(inputs, out1, ["public", "private"], seed=424242, fail_on_warning=True, max_concurrency=2)
    # Basic assertions
    assert (out1 / "run_summary.json").exists()
    # Public artifacts
    assert (out1 / "public_vendor_choice.csv").exists()
    assert (out1 / "public_tap_prices_per_model.csv").exists()
    assert (out1 / "public_tap_scenarios.csv").exists()
    assert (out1 / "public_tap_customers_by_month.csv").exists()
    assert (out1 / "public_tap_capacity_plan.csv").exists()
    # Private artifacts
    assert (out1 / "private_tap_economics.csv").exists()
    assert (out1 / "private_vendor_recommendation.csv").exists()
    assert (out1 / "private_tap_customers_by_month.csv").exists()
    # Consolidation
    assert (out1 / "consolidated_kpis.csv").exists()
    assert (out1 / "consolidated_summary.json").exists()
    assert (out1 / "consolidated_summary.md").exists()
    assert (out1 / "SHA256SUMS").exists()
