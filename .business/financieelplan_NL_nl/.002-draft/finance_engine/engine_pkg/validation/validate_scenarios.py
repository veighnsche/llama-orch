from __future__ import annotations

from pathlib import Path
import os

from .shared import FileReport, is_number
from ...io.loader import load_yaml


def validate(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "scenarios.yaml"
    fr = FileReport(name="scenarios.yaml", ok=True)
    if not p.exists():
        fr.ok = False
        fr.errors.append("scenarios.yaml missing (required)")
        return fr
    try:
        obj = load_yaml(p)
        driver = str((obj.get("driver") if isinstance(obj, dict) else None) or "tokens").lower()
        if driver == "funnel":
            # acquisition.yaml must be present
            acq = inputs_dir / "acquisition.yaml"
            if not acq.exists():
                fr.ok = False
                fr.errors.append("driver=funnel requires acquisition.yaml to be present")
            # monthly tokens optional in funnel mode
            fr.count = 1
        else:
            monthly = obj.get("monthly", {}) if isinstance(obj, dict) else {}
            for key in ("worst_m_tokens", "base_m_tokens", "best_m_tokens"):
                val = monthly.get(key)
                if not is_number(val) or float(val) < 0:
                    fr.ok = False
                    fr.errors.append(f"monthly.{key}: required numeric ≥ 0")
            fr.count = 3
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr
