from __future__ import annotations

from pathlib import Path
import os
from typing import Any, Dict
import datetime as dt

from .shared import FileReport, is_number
from ...io.loader import load_yaml


def validate(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "overrides.yaml"
    fr = FileReport(name="overrides.yaml", ok=True)
    strict = os.getenv("ENGINE_STRICT_VALIDATION", "0").lower() in {"1", "true", "yes"}
    if not p.exists():
        # Optional; used as temporary migration helper
        return fr
    try:
        obj = load_yaml(p)
        if not isinstance(obj, dict):
            fr.ok = False
            fr.errors.append("Top-level structure must be a mapping")
            return fr
        ov = obj.get("price_overrides", {})
        if not isinstance(ov, dict):
            fr.ok = False
            fr.errors.append("price_overrides: must be a mapping of SKU -> {unit_price_eur_per_1k_tokens}")
            return fr
        entries = []
        today = dt.date.today()
        for sku, node in ov.items():
            if not isinstance(sku, str) or not sku.strip():
                fr.ok = False
                fr.errors.append("price_overrides: empty SKU identifier")
                break
            if not isinstance(node, dict) or "unit_price_eur_per_1k_tokens" not in node:
                fr.ok = False
                fr.errors.append(f"price_overrides[{sku}]: must specify unit_price_eur_per_1k_tokens")
                break
            val = node.get("unit_price_eur_per_1k_tokens")
            if not is_number(val) or float(val) < 0:
                fr.ok = False
                fr.errors.append(f"price_overrides[{sku}].unit_price_eur_per_1k_tokens: must be numeric ≥ 0")
                break
            exp_str = node.get("expires_on")
            expired = False
            if isinstance(exp_str, str) and exp_str.strip():
                try:
                    exp = dt.date.fromisoformat(exp_str)
                    expired = exp < today
                except Exception:
                    fr.ok = False
                    fr.errors.append(f"price_overrides[{sku}].expires_on: invalid ISO date")
                    break
            if strict and expired:
                fr.ok = False
                fr.errors.append(f"price_overrides[{sku}] expired on {exp_str} (strict: error)")
                break
            entries.append({"sku": sku, "expires_on": exp_str, "expired": expired})
        fr.count = len(entries)
        fr.info["price_overrides_entries"] = entries
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr
