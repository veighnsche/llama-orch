"""Engine orchestrating financial model computation (Phase 2)
This module composes scenario, depreciation, loan, pnl, cashflow into one model dict.
No template rendering here; reports are handled by core.io_.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Tuple

from .money import D, ZERO, CENT
from .calendar import months_range, month_str, parse_month
from .scenario import apply_scenario
from .depreciation import build_depreciation
from .loan import build_amortization
from .pnl import compute_exploitatie
from .cashflow import compute_vat_accrual, schedule_vat_payment, build_liquidity


def build_month_axes(start_maand: str, months_count: int) -> Tuple[List[dt.date], List[str], Dict[str, Tuple[int, int]]]:
    start = parse_month(start_maand)
    dts = months_range(start, months_count)
    ym_list = [month_str(d) for d in dts]
    ym_to_y_m = {month_str(d): (d.year, d.month) for d in dts}
    return dts, ym_list, ym_to_y_m


def compute_model(cfg: Dict[str, Any], months_count: int, scenario: str, vat_period_flag: str) -> Dict[str, Any]:
    bedrijf = cfg['bedrijf']
    start_maand = bedrijf['start_maand']
    _dts, ym_list, ym_to_y_m = build_month_axes(start_maand, months_count)

    cfg_s = apply_scenario(cfg, scenario)

    # Depreciation
    inv = cfg_s.get('investeringen', [])
    dep_per_month, inv_items, total_invest = build_depreciation(inv, start_maand, ym_list)

    # Loans
    fin = cfg_s.get('financiering', {})
    eigen_inbreng = D(str(fin.get('eigen_inbreng', 0)))
    loans = fin.get('leningen', [])
    amort_rows, interest_pm, principal_pm = build_amortization(loans, ym_list)

    # Exploitatie inputs
    om = cfg_s.get('omzetmodel', {})
    omzet_pm = D(str(om.get('omzet_pm', 0)))
    cogs_pct_ratio = D(str(om.get('cogs_pct', 0))) / D('100')
    opex_map = {k: D(str(v)) for k, v in (om.get('opex_pm', {}) or {}).items()}
    dso_days = int(om.get('dso_dagen', 30))
    dpo_days = int(om.get('dpo_dagen', 14))
    season = om.get('seizoen', {}) or {}

    revenue, cogs, opex_lines, opex_total = compute_exploitatie(ym_list, omzet_pm, cogs_pct_ratio, opex_map, season)

    # VAT
    bel = cfg_s.get('belastingen', {})
    btw_pct_ratio = D(str(bel.get('btw_pct', 21))) / D('100')
    btw_vrij = bool(bel.get('btw_vrij', False))
    kor = bool(bel.get('kor', False))
    vat_model = bel.get('btw_model', 'omzet_enkel')
    vat_period = vat_period_flag or bel.get('vat_period', 'monthly')
    vat_accrual = compute_vat_accrual(ym_list, revenue, opex_total, vat_model, btw_pct_ratio, kor, btw_vrij)
    vat_payment = schedule_vat_payment(ym_list, ym_to_y_m, vat_accrual, vat_period)

    # Liquidity
    cash_begin, cash_end, cash_in_revenue, cash_out_cogs, inflow_other, opex_cash, lowest_cash, lowest_cash_month = build_liquidity(
        ym_list,
        revenue,
        cogs,
        opex_total,
        dep_per_month,
        interest_pm,
        principal_pm,
        eigen_inbreng,
        loans,
        dso_days,
        dpo_days,
        vat_payment,
    )

    # P&L aggregation for coverage
    avg_ebitda = ZERO
    if ym_list:
        total_ebitda = ZERO
        for ym in ym_list:
            marge = (revenue[ym] - cogs[ym]).quantize(CENT)
            ebitda = (marge - opex_total[ym]).quantize(CENT)
            total_ebitda += ebitda
        avg_ebitda = (total_ebitda / D(str(len(ym_list)))).quantize(CENT)
    avg_debt_service = ZERO
    if ym_list:
        total_ds = ZERO
        for ym in ym_list:
            total_ds += (interest_pm[ym] + principal_pm[ym])
        avg_debt_service = (total_ds / D(str(len(ym_list)))).quantize(CENT)
    coverage = (avg_ebitda / avg_debt_service) if avg_debt_service != 0 else D('0')

    return {
        'months': ym_list,
        'revenue': revenue,
        'cogs': cogs,
        'opex_lines': opex_lines,
        'opex_total': opex_total,
        'depreciation': dep_per_month,
        'interest': interest_pm,
        'principal': principal_pm,
        'cash_begin': cash_begin,
        'cash_end': cash_end,
        'cash_in_revenue': cash_in_revenue,
        'cash_out_cogs': cash_out_cogs,
        'inflow_other': inflow_other,
        'vat_payment': vat_payment,
        'amort_rows': amort_rows,
        'invest_items': inv_items,
        'total_invest': total_invest,
        'eigen_inbreng': eigen_inbreng,
        'lowest_cash': lowest_cash,
        'lowest_cash_month': lowest_cash_month,
        'avg_ebitda': avg_ebitda,
        'avg_debt_service': avg_debt_service,
        'coverage': coverage,
        'config': cfg_s,
    }
