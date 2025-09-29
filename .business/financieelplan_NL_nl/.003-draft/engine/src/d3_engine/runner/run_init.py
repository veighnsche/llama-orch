"""Run initialization utilities for the D3 engine runner."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterable

import yaml

from ..core.types import RunConfig
from ..core import logging as elog
from ..core import validator
from ..core import loader
from ..core import variables as vargrid
from ..core.artifacts import write_csv_header, append_csv_row


def build_run_config(
    inputs_dir: Path,
    out_dir: Path | None,
    pipelines: List[str] | None,
    seed: int | None,
    fail_on_warning: bool,
    max_concurrency: int | None,
) -> RunConfig:
    inputs_dir = Path(inputs_dir)
    # Defaults
    resolved_out = Path(out_dir) if out_dir else None
    resolved_pipelines: List[str] = [p for p in (pipelines or []) if str(p).strip() in ("public", "private")]

    # Fallbacks from simulation.yaml
    try:
        sim = yaml.safe_load((inputs_dir / "simulation.yaml").read_text()) or {}
    except Exception:
        sim = {}
    run = sim.get("run", {}) if isinstance(sim, dict) else {}

    if not resolved_pipelines:
        try:
            p_list = run.get("pipelines", [])
            if isinstance(p_list, list):
                resolved_pipelines = [p for p in p_list if str(p).strip() in ("public", "private")]
        except Exception:
            resolved_pipelines = []
        if not resolved_pipelines:
            resolved_pipelines = ["public", "private"]

    if resolved_out is None or str(resolved_out).strip() == "":
        try:
            out_override = run.get("output_dir")
            if isinstance(out_override, str) and out_override.strip():
                resolved_out = Path(out_override)
        except Exception:
            pass
    resolved_out = resolved_out or Path("./outputs")

    max_c = int(max_concurrency) if (max_concurrency and max_concurrency > 0) else 1

    return RunConfig(
        inputs_dir=inputs_dir,
        out_dir=resolved_out,
        pipelines=resolved_pipelines,
        seed=seed,
        fail_on_warning=bool(fail_on_warning),
        max_concurrency=max_c,
    )


def load_and_validate(inputs_dir: Path, fail_on_warning: bool) -> Dict[str, Any]:
    print(elog.jsonl("load_start", inputs=str(inputs_dir)))
    state = loader.load_all(inputs_dir)
    # Propagate fail_on_warning
    try:
        sim = state.setdefault("simulation", {})
        run = sim.setdefault("run", {})
        if fail_on_warning:
            run["fail_on_warning"] = True
    except Exception:
        pass
    print(elog.jsonl("load_done"))

    print(elog.jsonl("validate_start"))
    validator.validate(state)
    print(elog.jsonl("validate_done"))
    return state


def resolve_targets(state: Dict[str, Any]) -> Dict[str, Any]:
    """Extract acceptance targets and autoscaling target util from loaded state."""
    sim = state.get("simulation", {}) if isinstance(state, dict) else {}
    pub_op = state.get("operator", {}).get("public_tap", {}) if isinstance(state, dict) else {}

    target_utilization_pct = 75.0
    autoscaling_util_tolerance_pct = 25.0
    private_margin_threshold_pct = 20.0
    public_growth_min_mom_pct = None
    try:
        if isinstance(pub_op, dict):
            v = pub_op.get("autoscaling", {}).get("target_utilization_pct")
            if isinstance(v, (int, float)) and 1 <= float(v) <= 100:
                target_utilization_pct = float(v)
    except Exception:
        pass
    try:
        tol = (sim.get("targets", {}) if isinstance(sim, dict) else {})
        v = tol.get("autoscaling_util_tolerance_pct") if isinstance(tol, dict) else None
        if isinstance(v, (int, float)):
            autoscaling_util_tolerance_pct = float(v)
        pm = tol.get("private_margin_threshold_pct") if isinstance(tol, dict) else None
        if isinstance(pm, (int, float)):
            private_margin_threshold_pct = float(pm)
        pg = tol.get("public_growth_min_mom_pct") if isinstance(tol, dict) else None
        if isinstance(pg, (int, float)):
            public_growth_min_mom_pct = float(pg)
    except Exception:
        pass
    return {
        "target_utilization_pct": target_utilization_pct,
        "autoscaling_util_tolerance_pct": autoscaling_util_tolerance_pct,
        "private_margin_threshold_pct": private_margin_threshold_pct,
        **({"public_growth_min_mom_pct": public_growth_min_mom_pct} if public_growth_min_mom_pct is not None else {}),
    }


def plan_variables(state: Dict[str, Any]) -> Tuple[List[Tuple[str, float, float]], int, int, List[Dict[str, Any]], int]:
    """Collect variable metadata and sizes; runner decides on grid vs sampled.

    Returns: (random_specs, replicates, mc_count, all_vars, combos_potential_size)
    """
    try:
        vars_general = state.get("variables", {}).get("general", [])
        vars_public = state.get("variables", {}).get("public_tap", [])
        vars_private = state.get("variables", {}).get("private_tap", [])
        all_vars = list(vars_general) + list(vars_public) + list(vars_private)
        combos_size = int(vargrid.grid_size(all_vars))
    except Exception:
        all_vars = []
        combos_size = 1

    try:
        replicates = int(state.get("simulation", {}).get("run", {}).get("random_runs_per_simulation", 1))
        if replicates < 1:
            replicates = 1
    except Exception:
        replicates = 1

    try:
        mc_count = int(state.get("simulation", {}).get("stochastic", {}).get("simulations_per_run", 1))
        if mc_count < 1:
            mc_count = 1
    except Exception:
        mc_count = 1

    random_specs = vargrid.parse_random_specs(all_vars)
    return random_specs, replicates, mc_count, all_vars, combos_size


def write_variable_draws(out_dir: Path, all_vars: List[Dict[str, Any]], combos: List[Tuple[int, Dict[str, float]]], replicates: int) -> None:
    """Write variable_draws.csv (grid values as draws for v0)."""
    draws_path = Path(out_dir) / "variable_draws.csv"
    write_csv_header(draws_path, ["scope", "variable_id", "path", "grid_index", "replicate_index", "draw_value"])
    # Map path -> (scope, variable_id)
    path_meta: Dict[str, tuple[str, str]] = {}
    for row in all_vars:
        p = (row.get("path") or "").strip()
        if p and p not in path_meta:
            path_meta[p] = ((row.get("scope") or "").strip(), (row.get("variable_id") or "").strip())
    for gi, combo in combos:
        for ri in range(replicates):
            for pth, val in sorted(combo.items()):
                scope, varid = path_meta.get(pth, ("", ""))
                append_csv_row(draws_path, [scope, varid, pth, str(gi), str(ri), f"{val}"])
