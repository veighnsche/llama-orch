"""Autoscaling helpers (scaffold)."""
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple


def instances_needed(peak_tokens_per_hour: float, tps: float, target_utilization_pct: float) -> int:
    cap = max(tps, 1e-9) * 3600.0 * max(target_utilization_pct, 1.0) / 100.0
    return max(0, math.ceil(peak_tokens_per_hour / cap))


def cap_per_instance_tokens_per_hour(tps: float, target_utilization_pct: float) -> float:
    """Capacity per instance in tokens/hour at target utilization."""
    return max(tps, 1e-9) * 3600.0 * max(target_utilization_pct, 1.0) / 100.0


def planner_instances_needed(
    peak_tokens_per_hour: float,
    tps: float,
    target_utilization_pct: float,
    min_instances_per_model: int,
    max_instances_per_model: int,
) -> Tuple[int, bool, float]:
    """Deterministic planner result: (instances, capacity_violation, cap_per_instance)."""
    cap = cap_per_instance_tokens_per_hour(tps, target_utilization_pct)
    needed = max(0, math.ceil(peak_tokens_per_hour / cap))
    clamped = min(max(needed, max(0, min_instances_per_model)), max(1, max_instances_per_model))
    violation = needed > max_instances_per_model
    return clamped, violation, cap


@dataclass(frozen=True)
class ASGPolicy:
    evaluation_interval_s: int = 60
    scale_up_threshold_pct: float = 70.0
    scale_down_threshold_pct: float = 50.0
    scale_up_step_replicas: int = 1
    scale_down_step_replicas: int = 1
    stabilization_window_s: int = 300
    warmup_s: int = 120
    cooldown_s: int = 120
    min_instances_per_model: int = 0
    max_instances_per_model: int = 100


def simulate_autoscaler(
    demand_tokens_per_hour: List[float],
    tps: float,
    target_utilization_pct: float,
    policy: ASGPolicy,
    start_replicas: int | None = None,
) -> Dict[str, object]:
    """Deterministic autoscaler simulation (skeleton).

    Returns dict with keys: events (list), summary (dict).
    """
    cap_per_inst = cap_per_instance_tokens_per_hour(tps, target_utilization_pct)
    replicas = start_replicas if start_replicas is not None else max(policy.min_instances_per_model, 1)
    last_action_ts = -policy.cooldown_s
    ema = None
    alpha = 1.0 - math.exp(-policy.evaluation_interval_s / max(policy.stabilization_window_s, 1)) if policy.stabilization_window_s > 0 else 1.0
    warm_pool: List[Tuple[int, int]] = []  # (ts_available, replicas_added)

    events: List[Dict[str, object]] = []
    violations = 0
    util_sum = 0.0
    util_p95_bucket: List[float] = []

    for i, demand in enumerate(demand_tokens_per_hour):
        # Apply warmup: count only replicas whose warmup has elapsed
        ts = i * policy.evaluation_interval_s
        effective_replicas = replicas + sum(n for (t_ready, n) in list(warm_pool) if t_ready <= ts)
        warm_pool = [(t_ready, n) for (t_ready, n) in warm_pool if t_ready > ts]

        effective_capacity = max(effective_replicas, 0) * cap_per_inst
        observed_util = 0.0 if effective_capacity <= 0 else min(1.0, demand / effective_capacity)
        ema = observed_util if ema is None else (alpha * observed_util + (1 - alpha) * ema)
        util = ema
        util_sum += util
        util_p95_bucket.append(util)

        action = "hold"
        replicas_prev = replicas

        can_act = (ts - last_action_ts) >= policy.cooldown_s
        # Decide scale up/down using hysteresis
        if can_act and util > policy.scale_up_threshold_pct / 100.0 and replicas < policy.max_instances_per_model:
            step = policy.scale_up_step_replicas
            new_replicas = min(policy.max_instances_per_model, replicas + step)
            # warmup future capacity
            warm_pool.append((ts + policy.warmup_s, new_replicas - replicas))
            replicas = new_replicas
            last_action_ts = ts
            action = "scale_up"
        elif can_act and util < policy.scale_down_threshold_pct / 100.0 and replicas > policy.min_instances_per_model:
            step = policy.scale_down_step_replicas
            new_replicas = max(policy.min_instances_per_model, replicas - step)
            replicas = new_replicas
            last_action_ts = ts
            action = "scale_down"

        # SLA violation if demand exceeds effective capacity (ignores warming capacity)
        violation = demand > effective_capacity + 1e-9
        if violation:
            violations += 1

        events.append(
            {
                "timestamp_s": ts,
                "demand_tokens_per_hour": demand,
                "effective_capacity": effective_capacity,
                "replicas_prev": replicas_prev,
                "replicas_new": replicas,
                "reason": action,
                "util_pct": round(util * 100.0, 3),
            }
        )

    avg_util = util_sum / max(len(demand_tokens_per_hour), 1)
    util_p95 = sorted(util_p95_bucket)[int(0.95 * (len(util_p95_bucket) - 1))] if util_p95_bucket else 0.0
    summary = {
        "scale_events": sum(1 for e in events if e["reason"] in ("scale_up", "scale_down")),
        "avg_util_pct": round(avg_util * 100.0, 3),
        "p95_util_pct": round(util_p95 * 100.0, 3),
        "sla_violations": violations,
    }

    return {"events": events, "summary": summary}
