from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Any, Dict, List, Tuple

from finplan.common.money import D, CENT, ZERO


def sum_dec(vals) -> Decimal:
    s = ZERO
    for v in vals:
        s += D(str(v))
    return s.quantize(CENT)


def avg_dec(vals: List[Decimal]) -> Decimal:
    if not vals:
        return ZERO
    return (sum_dec(vals) / D(str(len(vals)))).quantize(CENT)


def pct(n: Decimal, d: Decimal) -> str:
    if d == 0:
        return "0%"
    return f"{(n / d * D('100')).quantize(D('0.01'))}%"


def ratio_pct(a: Decimal, b: Decimal) -> str:
    tot = a + b
    if tot == 0:
        return "0% / 0%"
    ap = (a / tot * D('100')).quantize(D('0.01'))
    bp = (b / tot * D('100')).quantize(D('0.01'))
    return f"{ap}% / {bp}%"


def best_worst(result_by_month: Dict[str, Decimal]) -> Tuple[Tuple[str, Decimal], Tuple[str, Decimal]]:
    best = ("", None)
    worst = ("", None)
    for ym, v in result_by_month.items():
        if best[1] is None or v > best[1]:
            best = (ym, v)
        if worst[1] is None or v < worst[1]:
            worst = (ym, v)
    return (best[0], best[1] or ZERO), (worst[0], worst[1] or ZERO)


def build_amort_index(amort_rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Decimal]]:
    idx: Dict[str, Dict[str, Decimal]] = defaultdict(lambda: {"rente_totaal": ZERO, "restschuld_einde": ZERO})
    for r in amort_rows:
        v = D(str(r.get('rente_pm', 0)))
        verstrekker = str(r.get('verstrekker', 'Onbekend'))
        idx[verstrekker]["rente_totaal"] += v
        idx[verstrekker]["restschuld_einde"] = D(str(r.get('restschuld', 0)))
    return idx


def breakeven_omzet_pm(revenue: Dict[str, Decimal], cogs: Dict[str, Decimal], opex_total: Dict[str, Decimal]) -> Decimal:
    rev_avg = avg_dec([revenue[m] for m in revenue])
    gm_ratio = ZERO
    if rev_avg > 0:
        gm_ratio = ((rev_avg - avg_dec([cogs[m] for m in cogs])) / rev_avg).quantize(D('0.0001'))
    fixed_costs = avg_dec([opex_total[m] for m in opex_total])
    if gm_ratio <= 0:
        return ZERO
    return (fixed_costs / gm_ratio).quantize(CENT)


def runway(cash_end: Dict[str, Decimal]) -> int:
    max_streak = 0
    cur = 0
    for ym in cash_end.keys():
        if cash_end[ym] >= ZERO:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0
    return max_streak
