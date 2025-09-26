from __future__ import annotations

from decimal import Decimal, getcontext, ROUND_HALF_UP
from typing import Union

# Deterministic, global Decimal context
_CTX = getcontext()
_CTX.prec = 28
_CTX.rounding = ROUND_HALF_UP

CENT = Decimal("0.01")

Number = Union[int, float, str, Decimal]


def as_decimal(x: Number) -> Decimal:
    """Convert to Decimal deterministically. Floats are converted via str()."""
    if isinstance(x, Decimal):
        return x
    if isinstance(x, (int,)):
        return Decimal(x)
    if isinstance(x, float):
        # Avoid binary float surprises
        return Decimal(str(x))
    if isinstance(x, str):
        # Allow comma decimal in user inputs
        xs = x.strip().replace("€", "").replace(" ", "").replace(".", "").replace(",", ".")
        try:
            return Decimal(xs)
        except Exception:
            # Fallback to original
            return Decimal(x)
    raise TypeError(f"Unsupported number type: {type(x)}")


def quantize_eur(d: Number) -> Decimal:
    return as_decimal(d).quantize(CENT)


def _thousands_dot(n: int) -> str:
    s = str(n)
    parts = []
    while s:
        parts.append(s[-3:])
        s = s[:-3]
    return ".".join(reversed(parts))


def format_eur_md(d: Number) -> str:
    """Return European currency formatting for Markdown: € 12.345,67"""
    q = quantize_eur(d)
    sign = "-" if q < 0 else ""
    q_abs = -q if q < 0 else q
    tup = str(q_abs).split(".")
    int_part = int(tup[0])
    frac_part = tup[1] if len(tup) > 1 else "00"
    if len(frac_part) < 2:
        frac_part = (frac_part + "00")[:2]
    return f"{sign}€ {_thousands_dot(int_part)},{frac_part}"


def format_decimal_csv(d: Number) -> str:
    """Dot-decimal string with 2 digits for CSV, independent of locale."""
    q = quantize_eur(d)
    s = f"{q:.2f}"
    # Ensure minus sign placement and dot decimal
    return s


def format_percent_md(pct: Number) -> str:
    q = as_decimal(pct).quantize(Decimal("0.01"))
    return f"{q}%"


def add(a: Number, b: Number) -> Decimal:
    return quantize_eur(as_decimal(a) + as_decimal(b))


def sub(a: Number, b: Number) -> Decimal:
    return quantize_eur(as_decimal(a) - as_decimal(b))


def mul(a: Number, b: Number) -> Decimal:
    return quantize_eur(as_decimal(a) * as_decimal(b))


def div(a: Number, b: Number) -> Decimal:
    d = as_decimal(b)
    if d == 0:
        return Decimal("0")
    return quantize_eur(as_decimal(a) / d)
