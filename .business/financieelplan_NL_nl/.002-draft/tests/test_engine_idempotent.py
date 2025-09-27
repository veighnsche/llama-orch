from __future__ import annotations

from finance_engine.config import OUTPUTS
from finance_engine.engine_pkg import run as engine_run


def test_engine_idempotent_two_runs_same_session():
    # First run
    rc1 = engine_run()
    assert rc1 == 0

    # Key artifacts exist
    plan = OUTPUTS / "financial_plan.md"
    assert plan.exists() and plan.stat().st_size > 0

    # Second run in the same test (no cleanup in-between)
    rc2 = engine_run()
    assert rc2 == 0

    # Artifacts still exist and are non-empty
    assert plan.exists() and plan.stat().st_size > 0
    for name in [
        "model_price_per_1m_tokens.csv",
        "public_tap_scenarios.csv",
        "private_tap_economics.csv",
        "break_even_targets.csv",
        "loan_schedule.csv",
        "vat_set_aside.csv",
    ]:
        p = OUTPUTS / name
        assert p.exists() and p.stat().st_size > 0
