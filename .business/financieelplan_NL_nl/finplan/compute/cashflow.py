"""Cashflow model: liquidity with DSO/DPO and VAT timing"""
from __future__ import annotations

from typing import Dict, List, Tuple, Any
from finplan.common.money import D, ZERO, CENT
from finplan.common.calendar import months_shift


def compute_cash_timing(
    ym_list: List[str],
    revenue: Dict[str, D],
    cogs: Dict[str, D],
    dso_days: int,
    dpo_days: int,
) -> Tuple[Dict[str, D], Dict[str, D]]:
    cash_in_revenue: Dict[str, D] = {ym: ZERO for ym in ym_list}
    cash_out_cogs: Dict[str, D] = {ym: ZERO for ym in ym_list}
    dso = months_shift(dso_days)
    dpo = months_shift(dpo_days)
    for i, ym in enumerate(ym_list):
        j = i + dso
        if j < len(ym_list):
            cash_in_revenue[ym_list[j]] += revenue[ym]
        j2 = i + dpo
        if j2 < len(ym_list):
            cash_out_cogs[ym_list[j2]] += cogs[ym]
    return cash_in_revenue, cash_out_cogs


def compute_vat_accrual(
    ym_list: List[str],
    revenue: Dict[str, D],
    opex_total: Dict[str, D],
    vat_model: str,
    btw_pct_ratio: D,
    kor: bool,
    btw_vrij: bool,
) -> Dict[str, D]:
    vat_accrual: Dict[str, D] = {ym: ZERO for ym in ym_list}
    if kor or btw_vrij:
        return vat_accrual
    if vat_model == 'omzet_enkel':
        for ym in ym_list:
            vat_accrual[ym] = (revenue[ym] * btw_pct_ratio).quantize(CENT)
    else:
        for ym in ym_list:
            base = (revenue[ym] * btw_pct_ratio)
            input_vat = (opex_total[ym] * btw_pct_ratio)
            vat_accrual[ym] = (base - input_vat).quantize(CENT)
    return vat_accrual


def schedule_vat_payment(
    ym_list: List[str],
    month_to_year_month: Dict[str, Tuple[int, int]],
    vat_accrual: Dict[str, D],
    vat_period: str,
) -> Dict[str, D]:
    vat_payment: Dict[str, D] = {ym: ZERO for ym in ym_list}
    if vat_period == 'monthly':
        for ym in ym_list:
            vat_payment[ym] = vat_accrual[ym]
        return vat_payment
    # quarterly: group by (year, quarter)
    group: Dict[Tuple[int, int], List[str]] = {}
    for ym in ym_list:
        y, m = month_to_year_month[ym]
        q = (y, (m - 1) // 3 + 1)
        group.setdefault(q, []).append(ym)
    for (_, _q), months in group.items():
        acc = ZERO
        for ym in months:
            acc += vat_accrual[ym]
        last_ym = months[-1]
        vat_payment[last_ym] = acc
    return vat_payment


def build_liquidity(
    ym_list: List[str],
    revenue: Dict[str, D],
    cogs: Dict[str, D],
    opex_total: Dict[str, D],
    dep: Dict[str, D],
    interest: Dict[str, D],
    principal: Dict[str, D],
    eigen_inbreng: D,
    loans: List[Dict[str, Any]],
    dso_days: int,
    dpo_days: int,
    vat_payment: Dict[str, D],
    invest_items: List[Dict[str, Any]] | None = None,
) -> Tuple[Dict[str, D], Dict[str, D], Dict[str, D], Dict[str, D], Dict[str, D], Dict[str, D], D, str]:
    cash_begin: Dict[str, D] = {}
    cash_end: Dict[str, D] = {}

    cash_in_revenue, cash_out_cogs = compute_cash_timing(ym_list, revenue, cogs, dso_days, dpo_days)

    inflow_other: Dict[str, D] = {ym: ZERO for ym in ym_list}
    if ym_list:
        inflow_other[ym_list[0]] += eigen_inbreng
        for ln in loans or []:
            inflow_other[ym_list[0]] += D(str(ln['hoofdsom']))

    opex_cash: Dict[str, D] = {ym: opex_total[ym] for ym in ym_list}

    # CAPEX payments at start months
    capex_out: Dict[str, D] = {ym: ZERO for ym in ym_list}
    for it in (invest_items or []):
        ym = str(it.get('start_maand'))
        if ym in capex_out:
            capex_out[ym] += D(str(it.get('bedrag', 0)))

    prev_end = ZERO
    lowest_cash = None
    lowest_cash_month = None
    for ym in ym_list:
        cash_begin[ym] = prev_end
        inflow = cash_in_revenue[ym] + inflow_other[ym]
        outflow = cash_out_cogs[ym] + opex_cash[ym] + vat_payment[ym] + interest[ym] + principal[ym] + capex_out[ym]
        end = (cash_begin[ym] + inflow - outflow).quantize(CENT)
        cash_end[ym] = end
        prev_end = end
        if lowest_cash is None or end < lowest_cash:
            lowest_cash = end
            lowest_cash_month = ym

    return (
        cash_begin,
        cash_end,
        cash_in_revenue,
        cash_out_cogs,
        inflow_other,
        opex_cash,
        lowest_cash or ZERO,
        lowest_cash_month or (ym_list[0] if ym_list else ''),
    )
