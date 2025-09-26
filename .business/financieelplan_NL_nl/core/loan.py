"""Loan amortization: annuity, grace, interest-only support"""
from __future__ import annotations

from typing import Dict, List, Tuple, Any
from .money import D, ZERO, CENT


def annuity_payment(P: D, r: D, n: int) -> D:
    if n <= 0:
        return ZERO
    if r == 0:
        return (P / D(str(n))).quantize(CENT)
    denom = D('1') - (D('1') + r) ** D(str(-n))
    A = (P * r) / denom
    return A.quantize(CENT)


def build_amortization(loans: List[Dict[str, Any]], ym_list: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, D], Dict[str, D]]:
    """Returns (schedule_rows, interest_pm, principal_pm)
    - loans: list with 'hoofdsom', 'rente_nominaal_jr_pct', 'looptijd_mnd', 'grace_mnd', 'alleen_rente_in_grace', 'verstrekker'
    - ym_list: list of months as YYYY-MM strings for the reporting horizon
    """
    schedule_rows: List[Dict[str, Any]] = []
    interest_pm: Dict[str, D] = {ym: ZERO for ym in ym_list}
    principal_pm: Dict[str, D] = {ym: ZERO for ym in ym_list}

    for ln in loans or []:
        P = D(str(ln['hoofdsom']))
        if P < 0:
            raise ValueError('lening.hoofdsom mag niet negatief zijn')
        r_year = D(str(ln['rente_nominaal_jr_pct'])) / D('100')
        r = (r_year / D('12'))
        r = r.quantize(D('0.0000001'))
        N = int(ln['looptijd_mnd'])
        grace = int(ln.get('grace_mnd', 0) or 0)
        only_interest = bool(ln.get('alleen_rente_in_grace', False))
        verstrekker = ln.get('verstrekker', 'Onbekend')

        balance = P
        # interest-only or partial annuity during grace
        A_full = annuity_payment(P, r, N)
        for i in range(min(grace, len(ym_list))):
            ym = ym_list[i]
            interest = (balance * r).quantize(CENT)
            principal = ZERO if only_interest else (A_full - interest).quantize(CENT)
            balance = (balance - principal).quantize(CENT)
            interest_pm[ym] += interest
            principal_pm[ym] += principal
            schedule_rows.append({
                'maand': ym,
                'verstrekker': verstrekker,
                'rente_pm': float(interest),
                'aflossing_pm': float(principal),
                'restschuld': float(balance),
            })

        remaining = max(0, N - grace)
        A = annuity_payment(balance, r, remaining) if remaining > 0 else ZERO
        for i in range(grace, min(N, len(ym_list))):
            ym = ym_list[i]
            interest = (balance * r).quantize(CENT)
            principal = (A - interest).quantize(CENT)
            if i == N - 1 or i == len(ym_list) - 1:
                principal = min(principal, balance)
            balance = (balance - principal).quantize(CENT)
            interest_pm[ym] += interest
            principal_pm[ym] += principal
            schedule_rows.append({
                'maand': ym,
                'verstrekker': verstrekker,
                'rente_pm': float(interest),
                'aflossing_pm': float(principal),
                'restschuld': float(balance),
            })

    return schedule_rows, interest_pm, principal_pm
