from __future__ import annotations

from pathlib import Path
from typing import Any

from .shared import FileReport, is_number
from ...io.loader import load_yaml


def _ok_mult(x: Any, *, allow_zero: bool = True) -> bool:
    try:
        v = float(x)
        if allow_zero:
            return v >= 0
        return v > 0
    except Exception:
        return False


keys = ("worst", "base", "best")
fields = ("budget_multiplier", "cvr_multiplier", "cac_multiplier")


def validate(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "funnel_overrides.yaml"
    fr = FileReport(name="funnel_overrides.yaml", ok=True)
    if not p.exists():
        # Optional file
        return fr
    try:
        obj = load_yaml(p)
        if not isinstance(obj, dict):
            fr.ok = False
            fr.errors.append("Top-level structure must be a mapping")
            return fr
        count = 0
        for k in keys:
            node = obj.get(k)
            if node is None:
                # not strictly required to have all keys; skip
                continue
            if not isinstance(node, dict):
                fr.ok = False
                fr.errors.append(f"{k}: must be a mapping")
                continue
            for f in fields:
                if f not in node:
                    fr.ok = False
                    fr.errors.append(f"{k}.{f}: required numeric multiplier")
                    continue
                v = node.get(f)
                if f == "cac_multiplier":
                    if not _ok_mult(v, allow_zero=False):
                        fr.ok = False
                        fr.errors.append(f"{k}.{f}: must be > 0")
                else:
                    if not _ok_mult(v, allow_zero=True):
                        fr.ok = False
                        fr.errors.append(f"{k}.{f}: must be ≥ 0")
            count += 1
        fr.count = count
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr
