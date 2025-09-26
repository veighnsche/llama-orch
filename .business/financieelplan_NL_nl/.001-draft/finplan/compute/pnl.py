"""Exploitatie (P&L accrual) calculations"""
from __future__ import annotations

from typing import Dict, Any, Tuple
from finplan.common.money import D, CENT, ZERO


def season_factor(season: Dict[str, Any], ym: str, key_suffix: str) -> D:
    m = (season or {}).get(ym, {})
    pct = D(str(m.get(key_suffix, 0)))
    return D('1') + (pct / D('100'))


def compute_exploitatie(
    ym_list: list[str],
    omzet_pm: D,
    cogs_pct_ratio: D,
    opex_map: Dict[str, D],
    season: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, D], Dict[str, D], Dict[str, Dict[str, D]], Dict[str, D]]:
    """Return (revenue, cogs, opex_lines, opex_total) per month.
    - omzet_pm: base revenue per month before season factors
    - cogs_pct_ratio: e.g., 0.35 for 35%
    - opex_map: {category: monthly amount}
    - season: { 'YYYY-MM': {'omzet_pm_pct': +x, 'opex_pm_pct': +y} }
    """
    revenue: Dict[str, D] = {}
    cogs: Dict[str, D] = {}
    opex_lines: Dict[str, Dict[str, D]] = {k: {} for k in (opex_map or {}).keys()}
    opex_total: Dict[str, D] = {}

    for ym in ym_list:
        rf = season_factor(season or {}, ym, 'omzet_pm_pct')
        of = season_factor(season or {}, ym, 'opex_pm_pct')
        rev = (omzet_pm * rf).quantize(CENT)
        revenue[ym] = rev
        c = (rev * cogs_pct_ratio).quantize(CENT)
        cogs[ym] = c
        ot = ZERO
        for k, base_v in (opex_map or {}).items():
            v = (base_v * of).quantize(CENT)
            opex_lines[k][ym] = v
            ot += v
        opex_total[ym] = ot

    return revenue, cogs, opex_lines, opex_total
