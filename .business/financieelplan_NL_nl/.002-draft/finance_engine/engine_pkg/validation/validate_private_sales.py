from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .shared import FileReport, is_number
from ...io.loader import load_yaml


def _pct_ok(x: Any) -> bool:
    try:
        v = float(x)
        return 0.0 <= v <= 100.0
    except Exception:
        return False


def validate(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "private_sales.yaml"
    fr = FileReport(name="private_sales.yaml", ok=True)
    if not p.exists():
        fr.ok = False
        fr.errors.append("private_sales.yaml missing (required)")
        return fr
    try:
        obj: Dict[str, Any] = load_yaml(p)
        if not isinstance(obj, dict):
            fr.ok = False
            fr.errors.append("Top-level structure must be a mapping")
            return fr
        req = [
            "leads_per_month",
            "win_rate_pct",
            "sales_cycle_months",
            "avg_commit_gpu_hours_per_month",
            "mgmt_fee_eur_per_month",
            "discount_pct_on_gpu_hour",
            "lead_conversion_from_public_pct",
        ]
        for k in req:
            v = obj.get(k)
            if k.endswith("_pct"):
                if not _pct_ok(v):
                    fr.ok = False
                    fr.errors.append(f"{k}: required 0–100")
            elif k == "sales_cycle_months":
                if not is_number(v) or int(float(v)) <= 0:
                    fr.ok = False
                    fr.errors.append("sales_cycle_months: required positive integer")
            else:
                if not is_number(v) or float(v) < 0:
                    fr.ok = False
                    fr.errors.append(f"{k}: required numeric ≥ 0")
        fr.count = len(req)
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr
