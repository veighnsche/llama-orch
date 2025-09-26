from __future__ import annotations

from typing import List, Dict


def _parse_ym(ym: str) -> (int, int):
    y, m = ym.split("-")
    return int(y), int(m)


def _to_ym(y: int, m: int) -> str:
    return f"{y:04d}-{m:02d}"


def _add_months(y: int, m: int, delta: int) -> (int, int):
    n = (y * 12 + (m - 1)) + delta
    yy = n // 12
    mm = (n % 12) + 1
    return yy, mm


def months_range(start_ym: str, horizon_months: int) -> List[str]:
    y, m = _parse_ym(start_ym)
    return [_to_ym(*_add_months(y, m, i)) for i in range(horizon_months)]


def shift_months(months: List[str], values: List, shift_days: int) -> List:
    """Shift a monthly series forward by approx days → full months (round half up)."""
    # Convert days to months (~30d per month), rounding half up
    months_shift = int((shift_days + 15) // 30) if shift_days >= 0 else -int((abs(shift_days) + 15) // 30)
    n = len(months)
    out = [0] * n
    for i, v in enumerate(values):
        j = i + months_shift
        if 0 <= j < n:
            out[j] = v
    return out


def is_quarter_end_index(idx: int) -> bool:
    # Every 3rd month (0-based) is a quarter end
    return (idx % 3) == 2


def vat_payment_flags(months: List[str], vat_period: str) -> List[bool]:
    if vat_period == "monthly":
        return [True] * len(months)
    elif vat_period == "quarterly":
        return [is_quarter_end_index(i) for i in range(len(months))]
    else:
        raise ValueError(f"Unsupported vat_period: {vat_period}")
