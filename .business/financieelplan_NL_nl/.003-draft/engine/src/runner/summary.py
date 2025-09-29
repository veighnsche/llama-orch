"""Run summary helpers."""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import yaml


def write_run_summary(out_dir: Path, summary: Dict, enabled: bool = True) -> List[str]:
    """Write run_summary.{json,md} if enabled. Return list of filenames written."""
    if not enabled:
        return []
    out_dir = Path(out_dir)
    (out_dir / "run_summary.json").write_text(yaml.safe_dump(summary, sort_keys=False))
    md_lines = [
        "# Run Summary",
        f"inputs: {summary.get('inputs','')}",
        f"outputs: {summary.get('outputs','')}",
        f"pipelines: {', '.join(summary.get('pipelines', []))}",
        f"seed: {summary.get('seed','')}",
        f"accepted: {summary.get('accepted','')}",
        "artifacts:",
    ] + [f"- {name}" for name in summary.get("artifacts", [])]
    (out_dir / "run_summary.md").write_text("\n".join(md_lines) + "\n")
    return ["run_summary.json", "run_summary.md"]
