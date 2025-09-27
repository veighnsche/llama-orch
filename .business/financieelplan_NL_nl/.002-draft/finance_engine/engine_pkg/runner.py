from __future__ import annotations

from typing import Any, Dict, Tuple
import math
import pandas as pd
import sys

from ..config import INPUTS, OUTPUTS, TEMPLATE_FILE, ENGINE_VERSION
from ..io.loader import load_yaml, read_csv
from ..io.writer import write_json, write_yaml, write_csv, ensure_dir
from ..validation.validate import validate_inputs
from ..utils.time import now_utc_iso
from ..utils.markdown import df_to_markdown_rows
from ..render.template import render_template_jinja

from ..compute.pricing import compute_model_economics
from ..compute.scenarios import compute_public_scenarios
from ..compute.private_tap import compute_private_tap_economics
from ..compute.break_even import compute_break_even
from ..compute.loans import Loan, flat_interest_schedule, loan_totals
from ..compute.vat import vat_examples
from ..charts.generate import (
    plot_model_margins,
    plot_public_scenarios,
    plot_break_even,
    plot_private_tap,
    plot_loan_balance,
)


def run() -> int:
    # Ensure outputs directory exists
    ensure_dir(OUTPUTS)

    # Load inputs
    config = load_yaml(INPUTS / "config.yaml")
    costs = load_yaml(INPUTS / "costs.yaml")
    lending = load_yaml(INPUTS / "lending_plan.yaml")
    extra = load_yaml(INPUTS / "extra.yaml") if (INPUTS / "extra.yaml").exists() else {}

    price_sheet = read_csv(INPUTS / "price_sheet.csv")
    gpu_df = read_csv(INPUTS / "gpu_rentals.csv")

    # Validate
    report = validate_inputs(config=config, costs=costs, lending=lending, price_sheet=price_sheet)
    write_json(OUTPUTS / "validation_report.json", report)
    if report.get("errors"):
        return 1

    # Minimal run summary
    run_summary: Dict[str, Any] = {
        "engine_version": ENGINE_VERSION,
        "run_at": now_utc_iso(),
        "notes": ["engine_pkg runner skeleton"],
    }
    write_json(OUTPUTS / "run_summary.json", run_summary)

    return 0
