from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .shared import FileReport, is_number
from ...io.loader import load_yaml


def _pct_ok(x: Any) -> bool:
    try:
        v = float(x)
        return 0.0 <= v <= 100.0
    except Exception:
        return False


def validate(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "seasonality.yaml"
    fr = FileReport(name="seasonality.yaml", ok=True)
    if not p.exists():
        fr.ok = False
        fr.errors.append("seasonality.yaml missing (required)")
        return fr
    try:
        obj: Dict[str, Any] = load_yaml(p)
        if not isinstance(obj, dict):
            fr.ok = False
            fr.errors.append("Top-level structure must be a mapping")
            return fr
        mm = obj.get("month_multipliers")
        if not isinstance(mm, list) or len(mm) != 12:
            fr.ok = False
            fr.errors.append("month_multipliers: required list of 12 numeric values")
        else:
            for i, x in enumerate(mm):
                if not is_number(x) or float(x) <= 0:
                    fr.ok = False
                    fr.errors.append(f"month_multipliers[{i}]: must be numeric > 0")
                    break
        dr = obj.get("diminishing_returns", {})
        if not isinstance(dr, dict):
            fr.ok = False
            fr.errors.append("diminishing_returns: required mapping")
        else:
            for k in ("cpc_slope_per_extra_1k_eur", "cvr_decay_per_budget_doubling_pct"):
                v = dr.get(k)
                if v is None or not is_number(v) or float(v) < 0:
                    fr.ok = False
                    fr.errors.append(f"diminishing_returns.{k}: required numeric ≥ 0")
        ev = obj.get("events")
        if ev is None or not isinstance(ev, list):
            fr.ok = False
            fr.errors.append("events: required list (can be empty)")
        fr.count = 12
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr
