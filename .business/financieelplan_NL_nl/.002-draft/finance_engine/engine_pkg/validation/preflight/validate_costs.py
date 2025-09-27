from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from .shared import FileReport, non_empty_string
from ....io.loader import load_yaml


def _walk_numeric_sum(d: Any, errs: List[str], path: str = "") -> float:
    total = 0.0
    if isinstance(d, dict):
        for k, v in d.items():
            if not non_empty_string(k):
                errs.append(f"empty key at {path}")
            total += _walk_numeric_sum(v, errs, f"{path}.{k}" if path else str(k))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            total += _walk_numeric_sum(v, errs, f"{path}[{i}]")
    else:
        try:
            val = float(d)
            if val < 0:
                errs.append(f"negative value at {path}")
            else:
                total += val
        except Exception:
            pass
    return total


def validate(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "costs.yaml"
    fr = FileReport(name="costs.yaml", ok=True)
    try:
        obj = load_yaml(p)
        if not isinstance(obj, dict):
            fr.ok = False
            fr.errors.append("Top-level structure must be a mapping")
            return fr
        errs: List[str] = []
        total = _walk_numeric_sum(obj, errs)
        if errs:
            fr.ok = False
            fr.errors.extend(sorted(set(errs)))
        if total == 0.0:
            fr.ok = False
            fr.errors.append("no numeric amounts found to compute a sum")
        fr.count = len(obj)
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr
