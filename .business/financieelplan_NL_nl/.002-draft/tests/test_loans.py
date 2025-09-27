from __future__ import annotations

from finance_engine.compute.loans import Loan, flat_interest_schedule, loan_totals


def test_loan_totals_and_schedule_flat_interest():
    loan = Loan(principal_eur=30000.0, annual_rate_pct=9.95, term_months=60)
    monthly, total_repay, total_interest = loan_totals(loan)
    assert monthly == 748.75
    assert total_repay == 44925.0
    assert total_interest == 14925.0

    rows = flat_interest_schedule(loan)
    assert len(rows) == 60
    # First row
    r1 = rows[0]
    assert r1["payment_eur"] == 748.75
    assert r1["balance_start_eur"] == 30000.0
    assert r1["balance_end_eur"] == 29500.0
    # Last row closes to zero
    r_last = rows[-1]
    assert r_last["balance_end_eur"] == 0.0
