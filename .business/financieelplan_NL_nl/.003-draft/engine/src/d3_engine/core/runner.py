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
from ..pipelines.public.demand import hourly_timeseries_uniform
from ..services.autoscaling import ASGPolicy, simulate_autoscaler


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
    autoscaling_summary = None
    if "public" in pipelines:
        print(elog.jsonl("pipeline_public_start"))
        artifacts += pub_art.write_all(out_dir)
        print(elog.jsonl("pipeline_public_done"))
        # Simulate autoscaler over a synthetic hourly demand profile (scaffold)
        # TODO: replace with real per-model series from pipelines once loader is implemented
        monthly_tokens = 1_000_000.0
        hours = 24 * 30
        series = hourly_timeseries_uniform(monthly_tokens, hours_in_month=hours, peak_factor=1.5, diurnal=True)
        # Assume a nominal throughput (tokens/s) and policy; these will be read from inputs later
        tps = 1_000.0  # tokens per second
        target_util = 75.0
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
        artifacts += prv_art.write_all(out_dir)
        print(elog.jsonl("pipeline_private_done"))

    # 3) Consolidation (TBD)
    print(elog.jsonl("consolidate_done"))

    # 4) Analysis (KPIs/percentiles/sensitivity)
    analysis = analyze({}, {"autoscaling_summary": autoscaling_summary} if autoscaling_summary else {})
    print(elog.jsonl("analysis_done"))

    # 5) Acceptance
    from . import acceptance as acc
    acceptance = acc.check_acceptance({"autoscaling_summary": autoscaling_summary}, {"target_utilization_pct": 75.0})
    print(elog.jsonl("acceptance_checked", accepted=acceptance.get("accepted")))

    return {
        "inputs": str(inputs_dir),
        "outputs": str(out_dir),
        "pipelines": pipelines,
        "seed": seed,
        "artifacts": artifacts,
        "analysis": analysis,
        "accepted": acceptance.get("accepted"),
    }
