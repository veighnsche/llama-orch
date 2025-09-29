"""SHA256SUMS writer for determinism checks."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional
import hashlib


def write_sha256sums(out_dir: Path, artifact_names: Iterable[str]) -> Optional[str]:
    out_dir = Path(out_dir)
    lines = []
    for name in sorted(set(artifact_names)):
        p = out_dir / name
        if not p.exists() or not p.is_file():
            continue
        h = hashlib.sha256()
        h.update(p.read_bytes())
        lines.append(f"{h.hexdigest()}  {name}")
    if not lines:
        return None
    sums_path = out_dir / "SHA256SUMS"
    sums_path.write_text("\n".join(lines) + "\n")
    return "SHA256SUMS"
