#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import sys
import subprocess
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUTPUTS = BASE / "outputs"
CHARTS = OUTPUTS / "charts"

MERMAID_BLOCK_RE = re.compile(r"```mermaid\s+([\s\S]*?)```", re.MULTILINE)


def pick_input_md() -> Path:
    fp = OUTPUTS / "financial_plan.md"
    if fp.exists():
        return fp
    tf = OUTPUTS / "template_filled.md"
    if tf.exists():
        return tf
    raise SystemExit("No markdown found. Run the engine to generate outputs/financial_plan.md or outputs/template_filled.md")


def prerender_mermaid(md_text: str) -> tuple[str, int]:
    CHARTS.mkdir(parents=True, exist_ok=True)
    count = 0

    def _replace(match: re.Match) -> str:
        nonlocal count
        diagram_src = match.group(1).strip()
        count += 1
        out_png = CHARTS / f"mermaid_{count}.png"
        # Write to temp file for mmdc input
        with tempfile.NamedTemporaryFile("w", suffix=".mmd", delete=False) as tmp:
            tmp.write(diagram_src)
            tmp_path = tmp.name
        try:
            # Use local devDependency via npx
            subprocess.check_call([
                "npx", "-y", "@mermaid-js/mermaid-cli",
                "-i", tmp_path,
                "-o", str(out_png),
                "--backgroundColor", "white",
            ], cwd=str(BASE))
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        # Return an image link relative to outputs/financial_plan.md location
        return f"![Mermaid diagram](charts/mermaid_{count}.png)"

    new_text = MERMAID_BLOCK_RE.sub(_replace, md_text)
    return new_text, count


def main(argv: list[str]) -> int:
    if len(argv) >= 2:
        in_path = Path(argv[1])
    else:
        in_path = pick_input_md()
    if len(argv) >= 3:
        out_path = Path(argv[2])
    else:
        out_path = OUTPUTS / "financial_plan.prerendered.md"

    md = in_path.read_text(encoding="utf-8")
    new_md, n = prerender_mermaid(md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(new_md, encoding="utf-8")
    print(f"Prerendered {n} mermaid diagram(s) -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
