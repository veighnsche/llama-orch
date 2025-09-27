from __future__ import annotations

import pytest

from finance_engine.compute.break_even import required_inflow, compute_break_even


def test_required_inflow_basic():
    # fixed 4000, margin 40%, marketing 15% -> inflow = 4000 / (0.4 - 0.15) = 16000
    assert required_inflow(4000.0, 0.40, 0.15) == 16000.0


def test_required_inflow_impossible():
    # margin_rate <= marketing_pct -> None
    assert required_inflow(4000.0, 0.10, 0.15) is None


def test_compute_break_even_returns_margin_value():
    be = compute_break_even(3000.0, 0.35, 0.15)
    assert be["required_inflow_eur"] == pytest.approx(15000.0, rel=1e-9, abs=1e-9)
    # required margin = margin_rate * inflow
    assert be["required_margin_eur"] == pytest.approx(0.35 * 15000.0, rel=1e-9, abs=1e-9)
