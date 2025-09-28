"""Engine runner (scaffold).
Coordinates: load → validate → (grid/replicates/MC TBD) → pipelines → artifacts → consolidate → acceptance → summary.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List

from . import loader, validator, logging as elog
from ..pipelines.public import artifacts as pub_art
from ..pipelines.private import artifacts as prv_art
from ..analysis import analyze


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def execute(inputs_dir: Path, out_dir: Path, pipelines: List[str], seed: int | None, fail_on_warning: bool) -> Dict:
    # 1) Load & validate
    print(elog.jsonl("load_start", inputs=str(inputs_dir)))
    state = loader.load_all(inputs_dir)
    print(elog.jsonl("load_done"))

    print(elog.jsonl("validate_start"))
    validator.validate(state)
    print(elog.jsonl("validate_done"))

    ensure_dir(out_dir)

    artifacts: List[str] = []

    # 2) Pipelines (placeholder generation of expected CSV headers)
    if "public" in pipelines:
        print(elog.jsonl("pipeline_public_start"))
        artifacts += pub_art.write_all(out_dir)
        print(elog.jsonl("pipeline_public_done"))

    if "private" in pipelines:
        print(elog.jsonl("pipeline_private_start"))
        artifacts += prv_art.write_all(out_dir)
        print(elog.jsonl("pipeline_private_done"))

    # 3) Consolidation (TBD)
    print(elog.jsonl("consolidate_done"))

    # 4) Analysis (KPIs/percentiles/sensitivity) (TBD)
    analysis = analyze({}, {})
    print(elog.jsonl("analysis_done"))

    # 5) Acceptance (TBD)
    print(elog.jsonl("acceptance_checked"))

    return {
        "inputs": str(inputs_dir),
        "outputs": str(out_dir),
        "pipelines": pipelines,
        "seed": seed,
        "artifacts": artifacts,
        "analysis": analysis,
    }
