"""Money & Decimal utilities (deterministic)
- Global Decimal context: ROUND_HALF_UP
- Helpers for currency and percentage formatting
"""
from __future__ import annotations

from decimal import Decimal, getcontext, ROUND_HALF_UP

# Deterministic Decimal context
CTX = getcontext()
CTX.prec = 28
CTX.rounding = ROUND_HALF_UP

D = Decimal
CENT = D("0.01")
HUNDRED = D("100")
ZERO = D("0")


def to_decimal(x) -> Decimal:
    if isinstance(x, Decimal):
        return x
    return D(str(x))


def money(x: Decimal) -> str:
    """Format to European currency: € 12.345,67"""
    x = to_decimal(x).quantize(CENT)
    neg = x < 0
    x = -x if neg else x
    s = f"{x:.2f}"
    integer, dec = s.split(".")
    groups = []
    while integer:
        groups.append(integer[-3:])
        integer = integer[:-3]
    res = "€ " + ".".join(groups[::-1]) + "," + dec
    return ("-" + res) if neg else res


def pct_from_ratio(r: Decimal) -> str:
    return f"{(to_decimal(r) * HUNDRED).quantize(CENT)}%"
