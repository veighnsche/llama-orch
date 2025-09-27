from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Dict, List
import datetime as dt

from .shared import FileReport, is_number
from ...io.loader import load_yaml


def validate(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "capacity_overrides.yaml"
    fr = FileReport(name="capacity_overrides.yaml", ok=True)
    strict = os.getenv("ENGINE_STRICT_VALIDATION", "0").lower() in {"1", "true", "yes"}
    if not p.exists():
        fr.ok = False
        fr.errors.append("capacity_overrides.yaml missing (required)")
        return fr
    try:
        obj = load_yaml(p)
        if not isinstance(obj, dict):
            fr.ok = False
            fr.errors.append("Top-level structure must be a mapping")
            return fr
        node = obj.get("capacity_overrides", {})
        if not isinstance(node, dict):
            fr.ok = False
            fr.errors.append("capacity_overrides: must be mapping of SKU -> {tps_override_tokens_per_sec?, preferred_gpu?, expires_on?}")
            return fr
        entries: List[Dict[str, Any]] = []
        today = dt.date.today()
        for sku, o in node.items():
            if not isinstance(sku, str) or not sku.strip():
                fr.ok = False
                fr.errors.append("capacity_overrides: empty SKU identifier")
                break
            if not isinstance(o, dict):
                fr.ok = False
                fr.errors.append(f"capacity_overrides[{sku}]: must be a mapping")
                break
            tps = o.get("tps_override_tokens_per_sec")
            if tps is not None and (not is_number(tps) or float(tps) < 0):
                fr.ok = False
                fr.errors.append(f"capacity_overrides[{sku}].tps_override_tokens_per_sec: must be numeric ≥ 0 if present")
                break
            preferred_gpu = o.get("preferred_gpu")
            if preferred_gpu is not None and (not isinstance(preferred_gpu, str) or not preferred_gpu.strip()):
                fr.ok = False
                fr.errors.append(f"capacity_overrides[{sku}].preferred_gpu: must be non-empty string if present")
                break
            exp_str = o.get("expires_on")
            expired = False
            if isinstance(exp_str, str) and exp_str.strip():
                try:
                    exp = dt.date.fromisoformat(exp_str)
                    expired = exp < today
                except Exception:
                    fr.ok = False
                    fr.errors.append(f"capacity_overrides[{sku}].expires_on: invalid ISO date")
                    break
            if strict and expired:
                fr.ok = False
                fr.errors.append(f"capacity_overrides[{sku}] expired on {exp_str} (strict: error)")
                break
            entries.append({
                "sku": sku,
                "tps_override_tokens_per_sec": float(tps) if is_number(tps) else None,
                "preferred_gpu": preferred_gpu,
                "expires_on": exp_str,
                "expired": expired,
            })
        fr.count = len(entries)
        fr.info["capacity_overrides_entries"] = entries
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr
