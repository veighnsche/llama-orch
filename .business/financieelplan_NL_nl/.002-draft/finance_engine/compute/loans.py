from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Loan:
    principal_eur: float
    annual_rate_pct: float
    term_months: int


def flat_interest_schedule(loan: Loan) -> List[dict]:
    """Flat interest schedule:
    - Total interest = P * (r/100) * (n/12)
    - Monthly interest = total_interest / n (constant)
    - Principal amortization per month = P / n (constant)
    - Monthly payment = principal_amort + monthly_interest (constant)
    """
    P = float(loan.principal_eur)
    r = float(loan.annual_rate_pct) / 100.0
    n = int(loan.term_months)
    total_interest = P * r * (n / 12.0)
    monthly_interest = total_interest / n
    principal_amort = P / n
    monthly_payment = principal_amort + monthly_interest

    rows: List[dict] = []
    balance = P
    for m in range(1, n + 1):
        interest_eur = monthly_interest
        principal_eur = principal_amort
        payment_eur = monthly_payment
        balance_end = max(0.0, balance - principal_eur)
        rows.append({
            "month": m,
            "balance_start_eur": round(balance, 2),
            "interest_eur": round(interest_eur, 2),
            "principal_eur": round(principal_eur, 2),
            "payment_eur": round(payment_eur, 2),
            "balance_end_eur": round(balance_end, 2),
        })
        balance = balance_end
    return rows


def loan_totals(loan: Loan) -> Tuple[float, float, float]:
    """Return (monthly_payment_eur, total_repayment_eur, total_interest_eur)."""
    P = float(loan.principal_eur)
    r = float(loan.annual_rate_pct) / 100.0
    n = int(loan.term_months)
    total_interest = P * r * (n / 12.0)
    monthly_payment = (P + total_interest) / n
    total_repayment = P + total_interest
    return round(monthly_payment, 2), round(total_repayment, 2), round(total_interest, 2)
