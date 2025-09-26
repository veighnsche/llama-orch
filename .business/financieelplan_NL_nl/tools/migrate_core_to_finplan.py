#!/usr/bin/env python3
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT / "core"
FIN = ROOT / "finplan"

# Map core files to their finplan destinations
MOVES = {
    CORE / "money.py": FIN / "common" / "money.py",
    CORE / "calendar.py": FIN / "common" / "calendar.py",
    CORE / "metrics.py": FIN / "common" / "metrics.py",
    CORE / "schema.py": FIN / "common" / "schema.py",
    CORE / "renderer.py": FIN / "render" / "renderer.py",
    CORE / "scan.py": FIN / "dev" / "scan.py",
    CORE / "io_.py": FIN / "fs.py",
    CORE / "loan.py": FIN / "compute" / "loan.py",
    CORE / "depreciation.py": FIN / "compute" / "depreciation.py",
    CORE / "cashflow.py": FIN / "compute" / "cashflow.py",
    CORE / "pnl.py": FIN / "compute" / "pnl.py",
    CORE / "scenario.py": FIN / "compute" / "scenario.py",
    CORE / "engine.py": FIN / "compute" / "engine.py",
    CORE / "mapping.py": FIN / "reporting" / "mapping.py",
    CORE / "context.py": FIN / "reporting" / "context.py",
}

REPLACEMENTS = [
    (r"from\s+\.money\s+import", "from finplan.common.money import"),
    (r"from\s+\.calendar\s+import", "from finplan.common.calendar import"),
    (r"from\s+\.metrics\s+import", "from finplan.common.metrics import"),
    (r"from\s+\.schema\s+import", "from finplan.common.schema import"),
    (r"from\s+\.loan\s+import", "from finplan.compute.loan import"),
    (r"from\s+\.depreciation\s+import", "from finplan.compute.depreciation import"),
    (r"from\s+\.cashflow\s+import", "from finplan.compute.cashflow import"),
    (r"from\s+\.pnl\s+import", "from finplan.compute.pnl import"),
    (r"from\s+\.scenario\s+import", "from finplan.compute.scenario import"),
    (r"from\s+\.renderer\s+import", "from finplan.render.renderer import"),
    (r"from\s+\.scan\s+import", "from finplan.dev.scan import"),
    (r"from\s+\.context\s+import", "from finplan.reporting.context import"),
]


def migrate() -> None:
    # Move files (overwrite finplan stubs if present)
    for src, dst in MOVES.items():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst.unlink()
        if not src.exists():
            raise SystemExit(f"Source missing: {src}")
        shutil.move(str(src), str(dst))
        print(f"Moved {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")

    # Rewrite imports inside moved files
    for dst in MOVES.values():
        text = dst.read_text(encoding="utf-8")
        for pat, repl in REPLACEMENTS:
            text = re.sub(pat, repl, text)
        dst.write_text(text, encoding="utf-8")
        print(f"Rewrote imports in {dst.relative_to(ROOT)}")

    # Remove old core directory
    if CORE.exists():
        shutil.rmtree(CORE)
        print("Removed core/ directory")


if __name__ == "__main__":
    migrate()
