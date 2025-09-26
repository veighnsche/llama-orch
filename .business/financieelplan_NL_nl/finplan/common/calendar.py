"""Calendar helpers: months, formatting, DSO/DPO shifts"""
from __future__ import annotations

import datetime as dt
from typing import List


def month_str(d: dt.date) -> str:
    return f"{d.year:04d}-{d.month:02d}"


def parse_month(s: str) -> dt.date:
    y, m = s.split('-')
    return dt.date(int(y), int(m), 1)


def add_months(d: dt.date, n: int) -> dt.date:
    y = d.year + (d.month - 1 + n) // 12
    m = (d.month - 1 + n) % 12 + 1
    return dt.date(y, m, 1)


def months_range(start: dt.date, n: int) -> List[dt.date]:
    return [add_months(start, i) for i in range(n)]


def months_shift(days: int) -> int:
    """Convert day terms to whole-month shifts using a simple 30-day month with mid-point rounding.
    E.g., 30 → +1 month, 14 → 0–1 month depending on midpoint rule → here: ((days+15)//30).
    """
    if days <= 0:
        return 0
    return max(0, int((days + 15) // 30))
