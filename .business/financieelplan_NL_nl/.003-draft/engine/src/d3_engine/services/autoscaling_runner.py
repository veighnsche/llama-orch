"""Autoscaling runner helpers: build policy and simulate+emit events."""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
from pathlib import Path

from .autoscaling import ASGPolicy, simulate_autoscaler
from ..runner.writers import write_scaling_events


def build_policy_from_public(pub_data: Dict[str, Any] | None) -> ASGPolicy:
    defaults = ASGPolicy()
    asg_cfg = {}
    try:
        asg_cfg = (pub_data or {}).get("autoscaling", {}) if isinstance(pub_data, dict) else {}
    except Exception:
        asg_cfg = {}
    try:
        return ASGPolicy(
            evaluation_interval_s=int(asg_cfg.get("evaluation_interval_s", defaults.evaluation_interval_s)),
            scale_up_threshold_pct=float(asg_cfg.get("scale_up_threshold_pct", defaults.scale_up_threshold_pct)),
            scale_down_threshold_pct=float(asg_cfg.get("scale_down_threshold_pct", defaults.scale_down_threshold_pct)),
            scale_up_step_replicas=int(asg_cfg.get("scale_up_step_replicas", defaults.scale_up_step_replicas)),
            scale_down_step_replicas=int(asg_cfg.get("scale_down_step_replicas", defaults.scale_down_step_replicas)),
            stabilization_window_s=int(asg_cfg.get("stabilization_window_s", defaults.stabilization_window_s)),
            warmup_s=int(asg_cfg.get("warmup_s", defaults.warmup_s)),
            cooldown_s=int(asg_cfg.get("cooldown_s", defaults.cooldown_s)),
            min_instances_per_model=int(asg_cfg.get("min_instances_per_model", defaults.min_instances_per_model)),
            max_instances_per_model=int(asg_cfg.get("max_instances_per_model", defaults.max_instances_per_model)),
        )
    except Exception:
        return defaults


def simulate_and_emit(
    out_dir: Path,
    demand_tokens_per_hour: List[float],
    tps: float,
    target_utilization_pct: float,
    policy: ASGPolicy,
    model: str = "demo_model",
    gpu: str = "demo_gpu",
) -> Tuple[Dict[str, Any], str]:
    """Run autoscaler simulation and emit events CSV. Returns (summary, artifact_name)."""
    sim = simulate_autoscaler(demand_tokens_per_hour, tps, target_utilization_pct, policy)
    events = sim.get("events", [])
    # Annotate events with model/gpu for CSV
    for e in events:
        e.setdefault("model", model)
        e.setdefault("gpu", gpu)
    artifact = write_scaling_events(out_dir, events)
    summary = sim.get("summary", {})
    return summary, artifact
