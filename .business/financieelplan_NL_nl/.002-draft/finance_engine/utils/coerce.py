from __future__ import annotations

from typing import Any, Mapping, Sequence


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def pct_to_fraction(pct_value: Any, default_pct: float = 0.0) -> float:
    return safe_float(pct_value, default_pct) / 100.0


def get(m: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    cur: Any = m
    try:
        for key in path:
            if not isinstance(cur, Mapping):
                return default
            cur = cur.get(key)
            if cur is None:
                return default
        return cur
    except Exception:
        return default


def get_float(m: Mapping[str, Any], path: Sequence[str], default: float = 0.0) -> float:
    return safe_float(get(m, path, default), default)
