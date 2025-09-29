from __future__ import annotations
from typing import Dict, Any, List


def build_loan_schedule(loan: Dict[str, Any], horizon: int) -> List[Dict[str, float]]:
    amt = float(loan.get("amount_eur", 0.0) or 0.0)
    term = int(loan.get("term_months", 0) or 0)
    rate_annual = float(loan.get("interest_rate_pct_flat", 0.0) or 0.0) / 100.0
    r = rate_annual / 12.0
    grace = int(loan.get("grace_period_months", 0) or 0)
    typ = str(loan.get("repayment_type", "annuity")).strip().lower()

    sched: List[Dict[str, float]] = []
    bal = amt
    pay = 0.0
    if typ == "annuity" and term > grace and r > 0:
        n = max(term - grace, 1)
        pay = bal * (r * (1 + r) ** n) / ((1 + r) ** n - 1)

    for m in range(horizon):
        interest = bal * r
        principal = 0.0
        if m < grace:
            payment = interest
        else:
            if typ == "annuity" and pay > 0:
                payment = pay
                principal = max(0.0, payment - interest)
                principal = min(principal, bal)
            elif typ == "flat" and term > 0:
                principal = amt / term
                payment = principal + interest
            elif typ == "bullet":
                principal = amt if m == term - 1 else 0.0
                payment = interest + (principal if principal > 0 else 0.0)
            else:
                payment = interest
                principal = 0.0
        bal_next = max(0.0, bal - principal)
        sched.append({
            "month": float(m),
            "principal_opening": bal,
            "interest": interest,
            "principal_repayment": principal,
            "principal_closing": bal_next,
            "payment": payment,
        })
        bal = bal_next
    return sched
