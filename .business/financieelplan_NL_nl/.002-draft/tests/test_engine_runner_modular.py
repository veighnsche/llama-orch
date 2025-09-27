from __future__ import annotations

from pathlib import Path
import json
import yaml
import re

from finance_engine.config import OUTPUTS
from finance_engine.engine_pkg import run as engine_run


def test_runner_produces_core_artifacts():
    # Execute the engine pipeline
    rc = engine_run()
    assert rc == 0, "engine run() should return 0 on success"

    # Core summary artifacts
    assert (OUTPUTS / "run_summary.json").exists(), "run_summary.json should be written"
    assert (OUTPUTS / "run_summary.md").exists(), "run_summary.md should be written"

    # Core CSV outputs
    expected_csvs = [
        "model_price_per_1m_tokens.csv",
        "public_tap_scenarios.csv",
        "private_tap_economics.csv",
        "break_even_targets.csv",
        "loan_schedule.csv",
        "vat_set_aside.csv",
    ]
    for name in expected_csvs:
        assert (OUTPUTS / name).exists(), f"{name} should be created"

    # Assumptions
    assumptions_path = OUTPUTS / "assumptions.yaml"
    assert assumptions_path.exists(), "assumptions.yaml should be created"
    data = yaml.safe_load(assumptions_path.read_text(encoding="utf-8"))
    # sanity keys
    assert "engine_version" in data
    assert "fx" in data and "eur_usd_rate" in data["fx"]
    assert "private" in data and "default_markup_over_cost_pct" in data["private"]


def test_runner_writes_financial_plan_and_charts():
    # Engine may have been run by previous test, but run again to be idempotent
    rc = engine_run()
    assert rc == 0

    # Financial plan
    plan = OUTPUTS / "financial_plan.md"
    assert plan.exists(), "financial_plan.md should be written"
    content = plan.read_text(encoding="utf-8")

    # Either rendered via Jinja or fallback; both are acceptable. Ensure has key sections.
    assert "Model Economics" in content, "Plan should include Model Economics section"

    # Charts
    charts = OUTPUTS / "charts"
    for chart in [
        "model_margins_per_1m.png",
        "public_scenarios_stack.png",
        "break_even.png",
        "private_tap_gpu_economics.png",
        "loan_balance_over_time.png",
    ]:
        assert (charts / chart).exists(), f"Chart should exist: {chart}"


def test_model_table_excludes_private_and_service_rows():
    # Ensure a run happened
    if not (OUTPUTS / "financial_plan.md").exists():
        assert engine_run() == 0

    content = (OUTPUTS / "financial_plan.md").read_text(encoding="utf-8")
    # In the plan, the model table should not include private/service SKUs
    forbidden = ["private_tap_", "priority_", "oss_support_"]
    for token in forbidden:
        assert token not in content, f"Model table should not contain {token} rows"
