from __future__ import annotations

from pathlib import Path
import os

from .shared import FileReport, is_number
from ...io.loader import load_yaml


def validate(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "scenarios.yaml"
    fr = FileReport(name="scenarios.yaml", ok=True)
    strict = os.getenv("ENGINE_STRICT_VALIDATION", "0").lower() in {"1", "true", "yes"}
    if not p.exists():
        if strict:
            fr.ok = False
            fr.errors.append("scenarios.yaml missing (strict: required)")
        else:
            fr.warnings.append("scenarios.yaml not present (optional in Step A)")
        return fr
    try:
        obj = load_yaml(p)
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
