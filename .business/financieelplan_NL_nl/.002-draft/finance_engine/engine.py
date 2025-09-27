from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json
import sys

from .config import INPUTS, OUTPUTS, TEMPLATE_FILE, ENGINE_VERSION
from .io.loader import load_yaml, read_csv
from .io.writer import write_json, ensure_dir
from .validation.validate import validate_inputs
from .utils.hashing import sha256_file
from .utils.time import now_utc_iso


def run() -> int:
    # Ensure outputs directory exists
    ensure_dir(OUTPUTS)

    # Load inputs (minimal for now; full implementation will expand)
    config = load_yaml(INPUTS / "config.yaml")
    costs = load_yaml(INPUTS / "costs.yaml")
    lending = load_yaml(INPUTS / "lending_plan.yaml")
    extra = load_yaml(INPUTS / "extra.yaml") if (INPUTS / "extra.yaml").exists() else {}

    price_sheet = read_csv(INPUTS / "price_sheet.csv")

    # Validate minimally
    report = validate_inputs(config=config, costs=costs, lending=lending, price_sheet=price_sheet)
    write_json(OUTPUTS / "validation_report.json", report)

    if report.get("errors"):
        # Non-zero exit expectation when errors exist
        return 1

    # Write a minimal run summary (full version will include inputs hashes, KPIs, etc.)
    try:
        inputs_hashes = {
            "config.yaml": sha256_file(INPUTS / "config.yaml"),
            "costs.yaml": sha256_file(INPUTS / "costs.yaml"),
            "lending_plan.yaml": sha256_file(INPUTS / "lending_plan.yaml"),
            "price_sheet.csv": sha256_file(INPUTS / "price_sheet.csv"),
        }
    except Exception:
        inputs_hashes = {}

    run_summary = {
        "engine_version": ENGINE_VERSION,
        "run_at": now_utc_iso(),
        "inputs_hashes": inputs_hashes,
        "notes": [
            "Scaffold run only. Computations and rendering will be added in the next step.",
        ],
    }
    write_json(OUTPUTS / "run_summary.json", run_summary)

    # Also write a minimal markdown summary for human-friendly review
    md_lines = [
        "# Run Summary",
        f"- Engine version: {ENGINE_VERSION}",
        f"- Run at (UTC): {run_summary['run_at']}",
        "",
        "## Input file hashes",
    ]
    for name, h in inputs_hashes.items():
        md_lines.append(f"- {name}: {h}")
    (OUTPUTS / "run_summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Write a placeholder filled template to verify pipeline (next step will replace with full renderer)
    if TEMPLATE_FILE.exists():
        (OUTPUTS / "template_filled.md").write_text(
            f"""# Template (placeholder)\n\nGenerated at: {run_summary['run_at']}\nEngine: {ENGINE_VERSION}\n\nFull rendering logic pending implementation.\n""",
            encoding="utf-8",
        )

    return 0


if __name__ == "__main__":
    sys.exit(run())
