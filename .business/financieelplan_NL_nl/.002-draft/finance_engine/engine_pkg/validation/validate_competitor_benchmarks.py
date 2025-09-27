from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .shared import FileReport, is_number, non_empty_string
from ...io.loader import load_yaml


def validate(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "competitor_benchmarks.yaml"
    fr = FileReport(name="competitor_benchmarks.yaml", ok=True)
    if not p.exists():
        fr.ok = False
        fr.errors.append("competitor_benchmarks.yaml missing (required)")
        return fr
    try:
        obj: Dict[str, Any] = load_yaml(p)
        if not isinstance(obj, dict):
            fr.ok = False
            fr.errors.append("Top-level structure must be a mapping")
            return fr
        per_sku = obj.get("per_sku")
        if not isinstance(per_sku, list) or len(per_sku) == 0:
            fr.ok = False
            fr.errors.append("per_sku: required non-empty list of {sku, eur_per_1k_tokens, source}")
        else:
            for i, row in enumerate(per_sku):
                if not isinstance(row, dict):
                    fr.ok = False
                    fr.errors.append(f"per_sku[{i}]: must be mapping")
                    break
                if not non_empty_string(row.get("sku")):
                    fr.ok = False
                    fr.errors.append(f"per_sku[{i}].sku: required non-empty string")
                    break
                if not is_number(row.get("eur_per_1k_tokens")) or float(row.get("eur_per_1k_tokens", 0)) <= 0:
                    fr.ok = False
                    fr.errors.append(f"per_sku[{i}].eur_per_1k_tokens: required numeric > 0")
                    break
                if not non_empty_string(row.get("source")):
                    fr.ok = False
                    fr.errors.append(f"per_sku[{i}].source: required non-empty string")
                    break
        pol = obj.get("policy")
        if not isinstance(pol, dict):
            fr.ok = False
            fr.errors.append("policy: required mapping {apply_competitor_caps: bool}")
        else:
            acc = pol.get("apply_competitor_caps")
            if not isinstance(acc, bool):
                fr.ok = False
                fr.errors.append("policy.apply_competitor_caps: required boolean")
        fr.count = len(per_sku) if isinstance(per_sku, list) else 0
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr
