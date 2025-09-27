from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .shared import FileReport, is_number
from ...io.loader import load_yaml


def validate(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "timeseries.yaml"
    fr = FileReport(name="timeseries.yaml", ok=True)
    if not p.exists():
        fr.ok = False
        fr.errors.append("timeseries.yaml missing (required)")
        return fr
    try:
        obj: Dict[str, Any] = load_yaml(p)
        if not isinstance(obj, dict):
            fr.ok = False
            fr.errors.append("Top-level structure must be a mapping")
            return fr
        # horizon must be 24 exactly
        h = obj.get("horizon_months")
        if int(float(h)) != 24:
            fr.ok = False
            fr.errors.append("horizon_months: required value 24")
        # reactivation & expansion
        for k in ("reactivation_pct", "expansion_arpu_pct"):
            v = obj.get(k)
            if v is None or not is_number(v):
                fr.ok = False
                fr.errors.append(f"{k}: required numeric")
            else:
                # allow any >=0; pct interpretation left to compute
                if float(v) < 0:
                    fr.ok = False
                    fr.errors.append(f"{k}: must be >= 0")
        cm = obj.get("cohort_model")
        if str(cm).strip().lower() != "simple_linear":
            fr.ok = False
            fr.errors.append("cohort_model: required 'simple_linear'")
        fr.count = 4
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr
