from __future__ import annotations

from finance_engine.config import OUTPUTS
from finance_engine.engine_pkg import run as engine_run


def test_template_full_sections_present():
    rc = engine_run()
    assert rc == 0
    plan = OUTPUTS / "financial_plan.md"
    assert plan.exists(), "financial_plan.md should exist"
    content = plan.read_text(encoding="utf-8")

    # Headline and executive summary
    assert "# Financial Plan" in content
    assert "## 0) Executive Summary" in content

    # Inputs sections
    assert "## 1) Inputs (Ground Truth)" in content
    assert "### 1.1 Prepaid Policy" in content
    assert "### 1.2 Catalog (Products Offered)" in content
    assert "### 1.3 Price Inputs" in content
    assert "### 1.4 Fixed Costs (Monthly)" in content
    assert "### 1.5 Tax & Billing" in content

    # Public tap economics and charts
    assert "## 2) Public Tap — Cost & Price per Model" in content
    assert "### 2.1 Model Economics (per 1M tokens)" in content
    assert "charts/model_margins_per_1m.png" in content

    # Scenarios
    assert "## 3) Public Tap — Monthly Projection Scenarios" in content
    assert "### 3.1 Scenario Table (per month)" in content
    assert "charts/public_scenarios_stack.png" in content
    assert "### 3.2 Break-even" in content
    assert "charts/break_even.png" in content

    # Private tap
    assert "## 4) Private Tap — Profitability Rules" in content
    assert "### 4.1 Table — GPU Economics (per hour)" in content
    assert "charts/private_tap_gpu_economics.png" in content or "charts/private-gpu-economics.png" in content

    # Combined projections
    assert "## 5) Worst/Best Case Projections" in content
    assert "### 5.1 Monthly Scenarios (snapshot)" in content
    assert "### 5.2 Yearly Projections (12 months)" in content
    assert "### 5.3 Loan-Term Projection (60 months)" in content

    # Loan schedule
    assert "## 6) Loan Schedule (60 Months)" in content
    assert "charts/loan_balance_over_time.png" in content

    # VAT and assurances
    assert "## 7) Taxes & VAT Set-Aside" in content
    assert "## 8) Assurances for Lender" in content

    # Appendices
    assert "## 9) Appendices" in content
