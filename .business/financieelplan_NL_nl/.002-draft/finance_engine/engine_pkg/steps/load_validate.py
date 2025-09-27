from __future__ import annotations

from typing import Any, Dict, Tuple, Optional

import pandas as pd

from ...config import INPUTS, OUTPUTS, ENGINE_VERSION
from ...io.loader import load_yaml, read_csv
from ...io.writer import write_json, write_text, ensure_dir
from ...utils.time import now_utc_iso
from ..ports import ValidatePort, get_default_validator


def load_inputs() -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    config = load_yaml(INPUTS / "config.yaml")
    costs = load_yaml(INPUTS / "costs.yaml")
    lending = load_yaml(INPUTS / "lending_plan.yaml")
    extra = load_yaml(INPUTS / "extra.yaml") if (INPUTS / "extra.yaml").exists() else {}
    price_sheet = read_csv(INPUTS / "price_sheet.csv")
    gpu_df = read_csv(INPUTS / "gpu_rentals.csv")
    return config, costs, lending, extra, price_sheet, gpu_df


def write_run_summary() -> Dict[str, Any]:
    ensure_dir(OUTPUTS)
    payload = {
        "engine_version": ENGINE_VERSION,
        "run_at": now_utc_iso(),
        "notes": ["engine_pkg orchestrator"],
    }
    write_json(OUTPUTS / "run_summary.json", payload)
    write_text(
        OUTPUTS / "run_summary.md",
        "\n".join([
            "# Run Summary",
            f"- Engine version: {ENGINE_VERSION}",
            f"- Run at (UTC): {payload['run_at']}",
        ]) + "\n",
    )
    return payload


def validate_and_write_report(
    config: Dict[str, Any],
    costs: Dict[str, Any],
    lending: Dict[str, Any],
    price_sheet: pd.DataFrame,
    *,
    validate_port: Optional[ValidatePort] = None,
) -> bool:
    validator: ValidatePort
    if validate_port is not None:
        validator = validate_port
    else:
        # Backward-compatible path: try to use runner.validate_inputs if tests monkeypatch it
        try:
            from .. import runner as runner_module  # type: ignore
            validator = getattr(runner_module, "validate_inputs")  # type: ignore
        except Exception:
            validator = get_default_validator()

    report = validator(config=config, costs=costs, lending=lending, price_sheet=price_sheet)
    write_json(OUTPUTS / "validation_report.json", report)
    return bool(report.get("errors"))
