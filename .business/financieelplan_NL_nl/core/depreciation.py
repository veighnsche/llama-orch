"""Straight-line depreciation with prorata start per item"""
from __future__ import annotations

from typing import Dict, List, Tuple, Any
from .money import D, ZERO, CENT
from .calendar import month_str, parse_month, add_months


def build_depreciation(investeringen: List[Dict[str, Any]], default_start_maand: str, months: List[str]) -> Tuple[Dict[str, D], List[Dict[str, Any]], D]:
    """Returns (dep_per_month, items_out, total_invest)
    - investeringen: list of items with bedrag, levensduur_mnd, start_maand(optional)
    - default_start_maand: company start month (YYYY-MM)
    - months: list of month strings over the horizon (YYYY-MM)
    """
    dep_per_month: Dict[str, D] = {ym: ZERO for ym in months}
    items_out: List[Dict[str, Any]] = []
    total_invest = ZERO

    for it in investeringen or []:
        bedrag = D(str(it['bedrag']))
        if bedrag < 0:
            raise ValueError('investering.bedrag mag niet negatief zijn')
        total_invest += bedrag
        life = int(it['levensduur_mnd'])
        if life <= 0:
            raise ValueError('levensduur_mnd moet > 0')
        start_s = it.get('start_maand') or default_start_maand
        start_i = parse_month(start_s)
        monthly = (bedrag / D(str(life))).quantize(CENT)
        items_out.append({
            'omschrijving': it['omschrijving'],
            'bedrag': float(bedrag),
            'levensduur_mnd': life,
            'start_maand': start_s,
            'afschrijving_pm': float(monthly),
        })
        for ym in months:
            mdate = parse_month(ym)
            if mdate >= start_i and (mdate < add_months(start_i, life)):
                dep_per_month[ym] += monthly

    return dep_per_month, items_out, total_invest
