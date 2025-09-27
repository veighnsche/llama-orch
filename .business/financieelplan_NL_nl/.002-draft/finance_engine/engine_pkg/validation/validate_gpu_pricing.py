from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Dict

from .shared import FileReport, is_number
from ...io.loader import load_yaml


def validate(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "gpu_pricing.yaml"
    fr = FileReport(name="gpu_pricing.yaml", ok=True)
    strict = os.getenv("ENGINE_STRICT_VALIDATION", "0").lower() in {"1", "true", "yes"}
    if not p.exists():
        if strict:
            fr.ok = False
            fr.errors.append("gpu_pricing.yaml missing (strict: required)")
        else:
            fr.warnings.append("gpu_pricing.yaml not present (optional)")
        return fr
    try:
        obj = load_yaml(p)
        if not isinstance(obj, dict):
            fr.ok = False
            fr.errors.append("Top-level structure must be a mapping")
            return fr
        node = obj.get("private_tap_markup_by_gpu", {})
        if not isinstance(node, dict):
            fr.ok = False
            fr.errors.append("private_tap_markup_by_gpu: must be a mapping of GPU -> percent")
            return fr
        for gpu, pct in node.items():
            if not isinstance(gpu, str) or gpu.strip() == "":
                fr.ok = False
                fr.errors.append("private_tap_markup_by_gpu: empty GPU name")
                break
            if not is_number(pct) or float(pct) < 0 or float(pct) > 100:
                fr.ok = False
                fr.errors.append(f"private_tap_markup_by_gpu[{gpu}]: must be numeric percent 0–100")
        fr.count = len(node)
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr
