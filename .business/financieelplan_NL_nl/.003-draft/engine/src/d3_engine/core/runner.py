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
from . import aggregate as agg
from ..pipelines.public.demand import hourly_timeseries_uniform
from ..services.autoscaling import ASGPolicy, simulate_autoscaler
import yaml
import hashlib
from . import rng
from . import variables as vargrid
from .artifacts import write_csv_header, append_csv_row, write_dict_rows


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def execute(inputs_dir: Path, out_dir: Path, pipelines: List[str], seed: int | None, fail_on_warning: bool, max_concurrency: int | None = None) -> Dict:
    # 1) Load & validate
    print(elog.jsonl("load_start", inputs=str(inputs_dir)))
    state = loader.load_all(inputs_dir)
    # Propagate CLI fail_on_warning into simulation.run
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

    ensure_dir(out_dir)

    artifacts: List[str] = []

    # Resolve master seed according to precedence (stochastic → run → operator meta)
    try:
        master_seed = rng.resolve_seed_from_state(state) if seed is None else int(seed)
    except Exception:
        master_seed = seed if seed is not None else None
    print(elog.jsonl("seed_resolved", seed=master_seed))
    if master_seed is None:
        raise RuntimeError("No random seed provided (stochastic/run/operator meta); see 33_engine_flow.md seed resolution")

    # Load acceptance targets from inputs
    target_utilization_pct = 75.0
    autoscaling_util_tolerance_pct = 25.0
    private_margin_threshold_pct = 20.0
    public_growth_min_mom_pct = None
    try:
        sim_p = inputs_dir / "simulation.yaml"
        sim_data = yaml.safe_load(sim_p.read_text()) or {}
        tol = (
            sim_data.get("targets", {})
            if isinstance(sim_data, dict) else {}
        )
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
    try:
        pub_p = inputs_dir / "operator" / "public_tap.yaml"
        pub_data = yaml.safe_load(pub_p.read_text()) or {}
        v = (
            pub_data.get("autoscaling", {})
            if isinstance(pub_data, dict) else {}
        ).get("target_utilization_pct")
        if isinstance(v, (int, float)) and 1 <= float(v) <= 100:
            target_utilization_pct = float(v)
    except Exception:
        pass

    # Build variables grid and replicates (v0: we still execute only the canonical job g0/r0)
    try:
        vars_general = state.get("variables", {}).get("general", [])
        vars_public = state.get("variables", {}).get("public_tap", [])
        vars_private = state.get("variables", {}).get("private_tap", [])
        all_vars = list(vars_general) + list(vars_public) + list(vars_private)
        combos = list(vargrid.iter_grid_combos(all_vars))
        grid_size = len(combos)
    except Exception:
        combos = [(0, {})]
        grid_size = 1
    print(elog.jsonl("grid_built", size=grid_size))

    # Variable draws transcript (optional but recommended)
    try:
        # Determine replicates count
        replicates = 1
        try:
            replicates = int(state.get("simulation", {}).get("run", {}).get("random_runs_per_simulation", 1))
            if replicates < 1:
                replicates = 1
        except Exception:
            replicates = 1
        draws_path = out_dir / "variable_draws.csv"
        write_csv_header(draws_path, [
            "scope", "variable_id", "path", "grid_index", "replicate_index", "draw_value"
        ])
        # Map path -> (scope, variable_id) for annotation
        path_meta = {}
        for row in all_vars:
            p = (row.get("path") or "").strip()
            if p and p not in path_meta:
                path_meta[p] = ((row.get("scope") or "").strip(), (row.get("variable_id") or "").strip())
        # Emit draws
        for gi, combo in combos:
            for ri in range(replicates):
                for pth, val in sorted(combo.items()):
                    scope, varid = path_meta.get(pth, ("", ""))
                    # For now, record the grid value as the draw (replicates do not perturb in v0)
                    append_csv_row(draws_path, [
                        scope, varid, pth, str(gi), str(ri), f"{val}"
                    ])
    except Exception:
        pass

    # 2) Pipelines over grid/replicates/MC (deterministic aggregation)
    autoscaling_summary = None
    # Prepare header caches for deterministic streaming writes
    public_headers: Dict[str, list[str]] = {}
    private_headers: Dict[str, list[str]] = {}
    # Random specs
    random_specs = vargrid.parse_random_specs(all_vars)
    # Replicates and MC nesting
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

    # Submit jobs
    job_idx = 0
    jobs: list[tuple[int, int, int, dict]] = []  # (gi, ri, mi, combo)
    for gi, combo in combos:
        for ri in range(replicates):
            for mi in range(mc_count):
                print(elog.jsonl("job_submitted", grid_index=gi, replicate_index=ri, mc_index=mi))
                jobs.append((gi, ri, mi, dict(combo)))
                job_idx += 1

    # Worker to compute rows for a single job
    def _compute_job(job: tuple[int, int, int, dict]):
        gi, ri, mi, combo = job
        random_overrides = vargrid.draw_randoms(random_specs, master_seed, gi, ri, mi)
        overrides = dict(combo)
        overrides.update(random_overrides)
        state_job = _with_overrides(state, overrides) if overrides else state
        pub_tables = pub_art.compute_rows(state_job) if "public" in pipelines else {}
        prv_tables = prv_art.compute_rows(state_job) if "private" in pipelines else {}
        return (gi, ri, mi, pub_tables, prv_tables)

    results: list[tuple[int, int, int, dict, dict]] = []
    workers = int(max_concurrency) if (max_concurrency and max_concurrency > 0) else 1
    if workers == 1:
        for job in jobs:
            print(elog.jsonl("job_started", grid_index=job[0], replicate_index=job[1], mc_index=job[2]))
            results.append(_compute_job(job))
            print(elog.jsonl("job_done", grid_index=job[0], replicate_index=job[1], mc_index=job[2]))
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as ex:
            future_map = {ex.submit(_compute_job, job): job for job in jobs}
            for fut in as_completed(future_map):
                gi, ri, mi, _, _ = future_map[fut]
                # Mark started/done when result arrives to keep logs concise; final outputs are deterministic
                print(elog.jsonl("job_started", grid_index=gi, replicate_index=ri, mc_index=mi))
                res = fut.result()
                results.append(res)
                print(elog.jsonl("job_done", grid_index=gi, replicate_index=ri, mc_index=mi))

    # Deterministic write-out (sorted)
    def _sorted_rows(name: str, rows: list[dict]) -> list[dict]:
        # Define stable sort keys per table name
        if name == "public_vendor_choice":
            return sorted(rows, key=lambda r: (r.get("model", ""), r.get("gpu", "")))
        if name == "public_tap_prices_per_model":
            return sorted(rows, key=lambda r: (r.get("model", ""), r.get("gpu", "")))
        if name == "public_tap_scenarios":
            return sorted(rows, key=lambda r: (r.get("scenario", ""), int(r.get("month", 0))))
        if name == "public_tap_customers_by_month":
            return sorted(rows, key=lambda r: (int(r.get("month", 0))))
        if name == "public_tap_capacity_plan":
            return sorted(rows, key=lambda r: (r.get("model", ""), r.get("gpu", "")))
        if name == "private_tap_economics":
            return sorted(rows, key=lambda r: (r.get("gpu", "")))
        if name == "private_vendor_recommendation":
            return sorted(rows, key=lambda r: (r.get("gpu", ""), r.get("provider", "")))
        if name == "private_tap_customers_by_month":
            return sorted(rows, key=lambda r: (int(r.get("month", 0))))
        return rows

    if "public" in pipelines:
        print(elog.jsonl("pipeline_public_start"))
        # Deterministic streaming write in sorted (gi, ri, mi) order for public tables
        for gi, ri, mi, pub_tables, _ in sorted(results, key=lambda t: (t[0], t[1], t[2])):
            if pub_tables:
                for name, (hdr, rows) in pub_tables.items():
                    if name not in public_headers:
                        public_headers[name] = hdr
                    rows_sorted = _sorted_rows(name, rows)
                    write_dict_rows(out_dir / f"{name}.csv", public_headers[name], rows_sorted)
        for name in sorted(public_headers.keys()):
            artifacts.append(f"{name}.csv")
        # Simulate autoscaler over a synthetic hourly demand profile (scaffold)
        # TODO: replace with real per-model series from pipelines once loader is implemented
        monthly_tokens = 1_000_000.0
        hours = 24 * 30
        series = hourly_timeseries_uniform(monthly_tokens, hours_in_month=hours, peak_factor=1.5, diurnal=True)
        # Assume a nominal throughput (tokens/s) and policy; these will be read from inputs later
        tps = 1_000.0  # tokens per second
        target_util = target_utilization_pct
        policy = ASGPolicy()
        sim = simulate_autoscaler(series, tps, target_util, policy)
        autoscaling_summary = sim.get("summary", {})
        # Append events to CSV
        events_path = out_dir / "public_tap_scaling_events.csv"
        if not events_path.exists():
            events_path.write_text(
                ",".join([
                    "timestamp_s","model","gpu","demand_tokens_per_hour","effective_capacity","replicas_prev","replicas_new","reason","util_pct"
                ]) + "\n"
            )
        with events_path.open("a") as f:
            for e in sim.get("events", []):
                row = [
                    str(e["timestamp_s"]),
                    "demo_model",
                    "demo_gpu",
                    f"{e['demand_tokens_per_hour']}",
                    f"{e['effective_capacity']}",
                    str(e["replicas_prev"]),
                    str(e["replicas_new"]),
                    str(e["reason"]),
                    f"{e['util_pct']}",
                ]
                f.write(",".join(row) + "\n")
        print(elog.jsonl("pipeline_public_done"))

    if "private" in pipelines:
        print(elog.jsonl("pipeline_private_start"))
        # Deterministic streaming write in sorted (gi, ri, mi) order for private tables
        for gi, ri, mi, _, prv_tables in sorted(results, key=lambda t: (t[0], t[1], t[2])):
            if prv_tables:
                for name, (hdr, rows) in prv_tables.items():
                    if name not in private_headers:
                        private_headers[name] = hdr
                    rows_sorted = _sorted_rows(name, rows)
                    write_dict_rows(out_dir / f"{name}.csv", private_headers[name], rows_sorted)
        for name in sorted(private_headers.keys()):
            artifacts.append(f"{name}.csv")
        print(elog.jsonl("pipeline_private_done"))

    # 3) Analysis (KPIs/percentiles/sensitivity)
    analysis_context = {
        "autoscaling_summary": autoscaling_summary,
        "out_dir": str(out_dir),
        "simulation": state.get("simulation", {}),
    }
    analysis = analyze({}, analysis_context)
    print(elog.jsonl("analysis_done"))

    # 4) Consolidation
    agg_ctx = {"autoscaling_summary": autoscaling_summary} if autoscaling_summary else {}
    if analysis and analysis.get("percentiles"):
        agg_ctx["percentiles"] = analysis.get("percentiles")
    if analysis and analysis.get("sensitivity"):
        agg_ctx["sensitivity"] = analysis.get("sensitivity")
    # Pass simulation and general finance context for overhead allocation
    agg_ctx["simulation"] = state.get("simulation", {})
    try:
        general_finance = state.get("operator", {}).get("general", {}).get("finance", {})
    except Exception:
        general_finance = {}
    agg_ctx["general_finance"] = general_finance
    written = agg.write_consolidated_outputs(out_dir, agg_ctx)
    artifacts += written
    print(elog.jsonl("aggregate_done"))
    print(elog.jsonl("consolidate_done"))

    # 5) Acceptance
    from . import acceptance as acc
    acceptance = acc.check_acceptance(
        {"autoscaling_summary": autoscaling_summary, "outputs_dir": str(out_dir)},
        {
            "target_utilization_pct": target_utilization_pct,
            "autoscaling_util_tolerance_pct": autoscaling_util_tolerance_pct,
            "private_margin_threshold_pct": private_margin_threshold_pct,
            **({"public_growth_min_mom_pct": public_growth_min_mom_pct} if public_growth_min_mom_pct is not None else {}),
        },
    )
    print(elog.jsonl("acceptance_checked", accepted=acceptance.get("accepted")))
    # Escalate policy violations when configured
    accepted_flag = bool(acceptance.get("accepted"))
    if not accepted_flag and fail_on_warning:
        raise RuntimeError("Acceptance checks failed and fail_on_warning is set")
    print(elog.jsonl("run_done"))

    # 6) Run summary (minimal; TODO: add input-hashes)
    # Compute input file hashes (SHA256) for determinism tracking
    input_hashes = []
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
    summary = {
        "inputs": str(inputs_dir),
        "outputs": str(out_dir),
        "pipelines": pipelines,
        "seed": master_seed,
        "artifacts": artifacts,
        "accepted": acceptance.get("accepted"),
        "input_hashes": input_hashes,
    }
    (out_dir / "run_summary.json").write_text(yaml.safe_dump(summary, sort_keys=False))
    md_lines = [
        "# Run Summary",
        f"inputs: {summary['inputs']}",
        f"outputs: {summary['outputs']}",
        f"pipelines: {', '.join(summary['pipelines'])}",
        f"seed: {summary['seed']}",
        f"accepted: {summary['accepted']}",
        "artifacts:",
    ] + [f"- {name}" for name in artifacts]
    (out_dir / "run_summary.md").write_text("\n".join(md_lines) + "\n")
    artifacts += ["run_summary.json", "run_summary.md"]

    # Write SHA256SUMS for determinism checks (golden tests)
    try:
        sums_path = out_dir / "SHA256SUMS"
        lines: List[str] = []
        for name in sorted(set(artifacts)):
            p = out_dir / name
            if not p.exists() or not p.is_file():
                continue
            h = hashlib.sha256()
            h.update(p.read_bytes())
            lines.append(f"{h.hexdigest()}  {name}")
        if lines:
            sums_path.write_text("\n".join(lines) + "\n")
            artifacts.append("SHA256SUMS")
    except Exception:
        pass

    return {
        "inputs": str(inputs_dir),
        "outputs": str(out_dir),
        "pipelines": pipelines,
        "seed": master_seed,
        "artifacts": artifacts,
        "analysis": analysis,
        "accepted": acceptance.get("accepted"),
    }
