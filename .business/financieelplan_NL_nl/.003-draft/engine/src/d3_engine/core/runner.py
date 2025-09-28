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

    # Load acceptance targets from inputs
    target_utilization_pct = 75.0
    autoscaling_util_tolerance_pct = 25.0
    private_margin_threshold_pct = 20.0
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

    # Grid built (single configuration in v0)
    print(elog.jsonl("grid_built", size=1))

    # 2) Pipelines (placeholder generation of expected CSV headers)
    autoscaling_summary = None
    if "public" in pipelines:
        print(elog.jsonl("pipeline_public_start"))
        print(elog.jsonl("job_submitted", grid_index=0, replicate_index=0))
        artifacts += pub_art.write_all(out_dir, state)
        print(elog.jsonl("pipeline_public_done"))
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

    if "private" in pipelines:
        print(elog.jsonl("pipeline_private_start"))
        print(elog.jsonl("job_submitted", grid_index=0, replicate_index=0))
        artifacts += prv_art.write_all(out_dir, state)
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
    written = agg.write_consolidated_outputs(out_dir, agg_ctx)
    artifacts += written
    print(elog.jsonl("aggregate_done"))
    print(elog.jsonl("consolidate_done"))

    # 5) Acceptance
    from . import acceptance as acc
    acceptance = acc.check_acceptance(
        {"autoscaling_summary": autoscaling_summary},
        {
            "target_utilization_pct": target_utilization_pct,
            "autoscaling_util_tolerance_pct": autoscaling_util_tolerance_pct,
            "private_margin_threshold_pct": private_margin_threshold_pct,
        },
    )
    print(elog.jsonl("acceptance_checked", accepted=acceptance.get("accepted")))
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
        "seed": seed,
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

    return {
        "inputs": str(inputs_dir),
        "outputs": str(out_dir),
        "pipelines": pipelines,
        "seed": seed,
        "artifacts": artifacts,
        "analysis": analysis,
        "accepted": acceptance.get("accepted"),
    }
