"""Engine runner (modular orchestration).
Coordinates: config → load+validate → variables/grid → execute jobs → materialize → autoscaling → analysis → consolidate → acceptance → summary.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import hashlib

from . import logging as elog
from . import rng
from .run_init import (
    build_run_config,
    load_and_validate,
    resolve_targets,
    plan_variables,
    write_variable_draws,
)
from .executor import execute_jobs
from .writers import CSVWriter, materialize_pipeline_tables
from .summary import write_run_summary
from .checksums import write_sha256sums
from ..pipelines import REGISTRY as PIPELINES_REGISTRY
from ..pipelines.public.demand import hourly_timeseries_uniform
from ..services.autoscaling_runner import build_policy_from_public, simulate_and_emit
from ..analysis import analyze
from ..aggregate import aggregator as agg


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def execute(inputs_dir: Path, out_dir: Path, pipelines: List[str], seed: int | None, fail_on_warning: bool, max_concurrency: int | None = None) -> Dict:
    # 1) Build config and load/validate
    cfg = build_run_config(inputs_dir, out_dir, pipelines, seed, fail_on_warning, max_concurrency)
    state = load_and_validate(cfg.inputs_dir, cfg.fail_on_warning)
    inputs_dir = cfg.inputs_dir
    out_dir = cfg.out_dir
    pipelines = cfg.pipelines
    ensure_dir(out_dir)

    artifacts: List[str] = []

    # Resolve master seed according to precedence (stochastic → run → operator meta)
    try:
        master_seed = rng.resolve_seed_from_state(state) if cfg.seed is None else int(cfg.seed)
    except Exception:
        master_seed = cfg.seed if cfg.seed is not None else None
    print(elog.jsonl("seed_resolved", seed=master_seed))
    if master_seed is None:
        raise RuntimeError("No random seed provided (stochastic/run/operator meta); see 33_engine_flow.md seed resolution")

    # 2) Targets & variables
    targets = resolve_targets(state)
    combos, random_specs, replicates, mc_count, all_vars = plan_variables(state)
    print(elog.jsonl("grid_built", size=len(combos)))
    try:
        write_variable_draws(out_dir, all_vars, combos, replicates)
    except Exception:
        pass

    # 3) Plan and execute jobs
    def _set_path(d: dict, dotted: str, value) -> None:
        cur = d
        parts = dotted.split(".")
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]
        cur[parts[-1]] = value

    def _with_overrides(base: dict, overrides: Dict[str, float]) -> dict:
        from copy import deepcopy
        new_state = deepcopy(base)
        for path, val in overrides.items():
            _set_path(new_state, path, val)
        return new_state

    jobs: list[tuple[int, int, int, dict]] = []
    for gi, combo in combos:
        for ri in range(replicates):
            for mi in range(mc_count):
                print(elog.jsonl("job_submitted", grid_index=gi, replicate_index=ri, mc_index=mi))
                jobs.append((gi, ri, mi, dict(combo)))

    def _compute_job(job: tuple[int, int, int, dict]):
        gi, ri, mi, combo = job
        random_overrides = vargrid.draw_randoms(random_specs, master_seed, gi, ri, mi)
        overrides = dict(combo)
        overrides.update(random_overrides)
        state_job = _with_overrides(state, overrides) if overrides else state
        pub_tables = PIPELINES_REGISTRY["public"](state_job) if "public" in pipelines else {}
        prv_tables = PIPELINES_REGISTRY["private"](state_job) if "private" in pipelines else {}
        return (gi, ri, mi, pub_tables, prv_tables)

    results: List[tuple[int, int, int, dict, dict]] = execute_jobs(jobs, _compute_job, cfg.max_concurrency)

    # 4) Materialize pipeline outputs
    if "public" in pipelines:
        print(elog.jsonl("pipeline_public_start"))
        csvw_pub = CSVWriter(out_dir)
        pub_results = [(gi, ri, mi, pub_tables) for gi, ri, mi, pub_tables, _ in results if pub_tables]
        artifacts += materialize_pipeline_tables(out_dir, pub_results, csvw_pub)
    if "private" in pipelines:
        print(elog.jsonl("pipeline_private_start"))
        csvw_prv = CSVWriter(out_dir)
        prv_results = [(gi, ri, mi, prv_tables) for gi, ri, mi, _, prv_tables in results if prv_tables]
        artifacts += materialize_pipeline_tables(out_dir, prv_results, csvw_prv)

    # 5) Autoscaling simulate (scaffold) and sanity log
    autoscaling_summary = None
    monthly_tokens = 1_000_000.0
    hours = 24 * 30
    series = hourly_timeseries_uniform(monthly_tokens, hours_in_month=hours, peak_factor=1.5, diurnal=True)
    tps = 1_000.0
    pub_data = state.get("operator", {}).get("public_tap", {}) if isinstance(state, dict) else {}
    policy = build_policy_from_public(pub_data)
    try:
        target_util = float(targets.get("target_utilization_pct", 75.0))
        tol = float(targets.get("autoscaling_util_tolerance_pct", 25.0))
    except Exception:
        target_util, tol = 75.0, 25.0
    checks = {
        "thresholds_order": float(policy.scale_down_threshold_pct) < float(policy.scale_up_threshold_pct),
        "scale_up_ge_target": float(policy.scale_up_threshold_pct) >= float(target_util),
        "scale_down_le_target": float(policy.scale_down_threshold_pct) <= float(target_util),
        "target_bracket_within_range": (float(target_util) - tol) >= 0.0 and (float(target_util) + tol) <= 100.0,
        "min_le_max": int(policy.min_instances_per_model) <= int(policy.max_instances_per_model),
    }
    print(elog.jsonl("autoscaling_sanity", ok=all(checks.values()), target_util_pct=target_util, tolerance_pct=tol, **checks))
    autoscaling_summary, asg_art = simulate_and_emit(out_dir, series, tps, target_util, policy)
    artifacts += [asg_art]

    # 6) Analysis
    analysis_context = {
        "autoscaling_summary": autoscaling_summary,
        "out_dir": str(out_dir),
        "simulation": state.get("simulation", {}),
    }
    analysis = analyze({}, analysis_context)
    print(elog.jsonl("analysis_done"))

    # 7) Consolidation
    agg_ctx = {"autoscaling_summary": autoscaling_summary} if autoscaling_summary else {}
    if analysis and analysis.get("percentiles"):
        agg_ctx["percentiles"] = analysis.get("percentiles")
    if analysis and analysis.get("sensitivity"):
        agg_ctx["sensitivity"] = analysis.get("sensitivity")
    agg_ctx["simulation"] = state.get("simulation", {})
    try:
        operator_general = state.get("operator", {}).get("general", {})
    except Exception:
        operator_general = {}
    agg_ctx["operator_general"] = operator_general
    written = agg.write_consolidated_outputs(out_dir, agg_ctx)
    artifacts += written
    print(elog.jsonl("aggregate_done"))
    print(elog.jsonl("consolidate_done"))

    # 8) Acceptance
    from . import acceptance as acc
    acceptance = acc.check_acceptance(
        {"autoscaling_summary": autoscaling_summary, "outputs_dir": str(out_dir)},
        targets,
    )
    print(elog.jsonl("acceptance_checked", accepted=acceptance.get("accepted")))
    if not bool(acceptance.get("accepted")) and cfg.fail_on_warning:
        raise RuntimeError("Acceptance checks failed and fail_on_warning is set")
    print(elog.jsonl("run_done"))

    # 9) Determinism: input hashes
    input_hashes: List[Dict[str, str]] = []
    try:
        for p in sorted(inputs_dir.rglob("*")):
            if p.is_file():
                h = hashlib.sha256()
                try:
                    h.update(p.read_bytes())
                    input_hashes.append({"path": str(p.relative_to(inputs_dir)), "sha256": h.hexdigest()})
                except Exception:
                    continue
    except Exception:
        pass

    # 10) Summary (honor logging.write_run_summary from state)
    write_flag = True
    try:
        write_flag = bool(state.get("simulation", {}).get("logging", {}).get("write_run_summary", True))
    except Exception:
        write_flag = True
    summary = {
        "inputs": str(inputs_dir),
        "outputs": str(out_dir),
        "pipelines": pipelines,
        "seed": master_seed,
        "artifacts": artifacts,
        "accepted": acceptance.get("accepted"),
        "input_hashes": input_hashes,
    }
    artifacts += write_run_summary(out_dir, summary, enabled=write_flag)

    # 11) SHA256SUMS
    sums_name = write_sha256sums(out_dir, artifacts)
    if sums_name:
        artifacts.append(sums_name)

    return {
        "inputs": str(inputs_dir),
        "outputs": str(out_dir),
        "pipelines": pipelines,
        "seed": master_seed,
        "artifacts": artifacts,
        "analysis": analysis,
        "accepted": acceptance.get("accepted"),
    }
