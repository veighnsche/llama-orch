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
from .loan import build_amortization, annuity_payment
from .pnl import compute_exploitatie
from .cashflow import compute_vat_accrual, schedule_vat_payment, build_liquidity


def build_month_axes(start_maand: str, months_count: int) -> Tuple[List[dt.date], List[str], Dict[str, Tuple[int, int]]]:
    start = parse_month(start_maand)
    dts = months_range(start, months_count)
    ym_list = [month_str(d) for d in dts]
    ym_to_y_m = {month_str(d): (d.year, d.month) for d in dts}
    return dts, ym_list, ym_to_y_m


def compute_model(cfg: Dict[str, Any], months_count: int, scenario: str, vat_period_flag: str) -> Dict[str, Any]:
    bedrijf = cfg.get('bedrijf', {})
    start_maand = bedrijf.get('start_maand', '2025-01')
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

    # Exploitatie inputs from omzetstromen + opex_vast_pm
    streams = cfg_s.get('omzetstromen', []) or []
    # Build revenue and cogs per month
    revenue: Dict[str, D] = {ym: ZERO for ym in ym_list}
    cogs: Dict[str, D] = {ym: ZERO for ym in ym_list}
    for s in streams:
        prijs = D(str(s.get('prijs', 0)))
        var_eh = D(str(s.get('var_kosten_per_eenheid', 0)))
        vols = s.get('volume_pm') or [0] * len(ym_list)
        for i, ym in enumerate(ym_list):
            vol = D(str(vols[i])) if i < len(vols) else D('0')
            revenue[ym] += (prijs * vol).quantize(CENT)
            cogs[ym] += (var_eh * vol).quantize(CENT)

    # OPEX lines
    opex_src = cfg_s.get('opex_vast_pm', {}) or {}
    opex_lines: Dict[str, Dict[str, D]] = {}
    opex_total: Dict[str, D] = {ym: ZERO for ym in ym_list}
    # Personeel list
    personeel_list = (opex_src.get('personeel') or []) if isinstance(opex_src.get('personeel'), list) else []
    personeel_pm = ZERO
    for p in personeel_list:
        personeel_pm += D(str(p.get('bruto_pm', 0)))
    if personeel_pm != ZERO:
        opex_lines['personeel'] = {ym: personeel_pm for ym in ym_list}
        for ym in ym_list:
            opex_total[ym] += personeel_pm
    # Scalar categories
    for cat in ['marketing', 'software', 'huisvesting', 'overig']:
        val = D(str(opex_src.get(cat, 0)))
        if val != ZERO:
            opex_lines[cat] = {ym: val for ym in ym_list}
            for ym in ym_list:
                opex_total[ym] += val

    # DSO/DPO from werkkapitaal
    wc = cfg_s.get('werkkapitaal', {}) or {}
    dso_days = int(wc.get('dso_dagen', 30))
    dpo_days = int(wc.get('dpo_dagen', 14))

    # VAT
    bel = cfg_s.get('btw', {}) or {}
    btw_pct_ratio = D(str(bel.get('btw_pct', 21))) / D('100')
    btw_vrij = bool(bel.get('btw_vrij', False))
    kor = bool(bel.get('kor', False))
    vat_model = bel.get('model', 'omzet_enkel')
    vat_period = vat_period_flag or cfg_s.get('vat_period', 'monthly')
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
        inv_items,
    )

    # P&L aggregation and DSCR
    avg_ebitda = ZERO
    ebitda_pm: Dict[str, D] = {}
    dscr_pm: Dict[str, D] = {}
    if ym_list:
        total_ebitda = ZERO
        for ym in ym_list:
            marge = (revenue[ym] - cogs[ym]).quantize(CENT)
            ebitda = (marge - opex_total[ym]).quantize(CENT)
            ebitda_pm[ym] = ebitda
            total_ebitda += ebitda
            debt_svc = (interest_pm[ym] + principal_pm[ym])
            dscr_pm[ym] = (ebitda / debt_svc) if debt_svc != 0 else D('0')
        avg_ebitda = (total_ebitda / D(str(len(ym_list)))).quantize(CENT)
    avg_debt_service = ZERO
    if ym_list:
        total_ds = ZERO
        for ym in ym_list:
            total_ds += (interest_pm[ym] + principal_pm[ym])
        avg_debt_service = (total_ds / D(str(len(ym_list)))).quantize(CENT)
    coverage = (avg_ebitda / avg_debt_service) if avg_debt_service != 0 else D('0')

    # DSCR metrics
    dscr_min = D('0')
    dscr_maand_min = ''
    dscr_below_1_count = 0
    for ym in ym_list:
        v = dscr_pm.get(ym, D('0'))
        if ym == ym_list[0] or v < dscr_min:
            dscr_min = v
            dscr_maand_min = ym
        if v < D('1'):
            dscr_below_1_count += 1

    # Runway (max consecutive months cash_end >= 0)
    max_streak = 0
    cur = 0
    for ym in ym_list:
        if cash_end[ym] >= ZERO:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0

    return {
        'months': ym_list,
        'revenue': revenue,
        'cogs': cogs,
        'opex_lines': opex_lines,
        'opex_total': opex_total,
        'depreciation': dep_per_month,
        'interest': interest_pm,
        'principal': principal_pm,
        'ebitda': ebitda_pm,
        'dscr': dscr_pm,
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
        'dscr_min': dscr_min,
        'dscr_maand_min': dscr_maand_min,
        'dscr_below_1_count': dscr_below_1_count,
        'runway_maanden_base': max_streak,
        'config': cfg_s,
    }
