from __future__ import annotations

from pathlib import Path

from .shared import FileReport
from ....io.loader import load_yaml


def validate(inputs_dir: Path) -> FileReport:
    p = inputs_dir / "extra.yaml"
    fr = FileReport(name="extra.yaml", ok=True)
    if not p.exists():
        fr.warnings.append("extra.yaml not present (Step A: optional)")
        return fr
    try:
        _ = load_yaml(p)
    except Exception as e:
        fr.ok = False
        fr.errors.append(f"parse error: {e}")
    return fr
