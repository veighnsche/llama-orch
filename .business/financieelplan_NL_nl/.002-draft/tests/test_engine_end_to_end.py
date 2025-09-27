from __future__ import annotations

from pathlib import Path

from finance_engine.engine_pkg import run
from finance_engine.config import OUTPUTS


def test_engine_generates_markdown_and_charts(tmp_path):
    rc = run()
    assert rc == 0
    # Core outputs
    assert (OUTPUTS / "model_price_per_1m_tokens.csv").exists()
    assert (OUTPUTS / "public_tap_scenarios.csv").exists()
    assert (OUTPUTS / "private_tap_economics.csv").exists()
    assert (OUTPUTS / "break_even_targets.csv").exists()
    assert (OUTPUTS / "loan_schedule.csv").exists()
    assert (OUTPUTS / "vat_set_aside.csv").exists()
    # Charts
    charts = OUTPUTS / "charts"
    assert (charts / "model_margins_per_1m.png").exists()
    assert (charts / "public_scenarios_stack.png").exists()
    assert (charts / "break_even.png").exists()
    assert (charts / "private_tap_gpu_economics.png").exists()
    assert (charts / "loan_balance_over_time.png").exists()
    # Markdown plan (either Jinja or fallback must exist)
    assert (OUTPUTS / "financial_plan.md").exists()
