"""Acceptance checks (scaffold).

Responsibilities (spec refs: 40_testing.md §5, 20_simulations.md §7.1):
- Monotone growth (public/private) and optional min MoM growth
- Private margin threshold per GPU-class
- Capacity violations (instances_needed ≤ autoscaling.max_instances_per_model)

Return structured results to be embedded in run_summary and used for exit policy.
"""
from __future__ import annotations
from typing import Dict, Any


def check_acceptance(tables: Dict[str, Any], targets: Dict[str, Any]) -> Dict[str, Any]:
    """Acceptance focusing on autoscaling KPIs until other tables are wired.

    Conditions:
    - p95_util_pct within tolerance around target_utilization_pct
    - sla_violations == 0
    """
    autos = tables.get("autoscaling_summary") or {}
    target_util = float(targets.get("target_utilization_pct", 75.0))
    tol = float(targets.get("autoscaling_util_tolerance_pct", 25.0))  # wide default during scaffold

    p95 = autos.get("p95_util_pct")
    violations = autos.get("sla_violations")

    p95_ok = False if p95 is None else (target_util - tol <= float(p95) <= target_util + tol)
    no_violations = False if violations is None else (int(violations) == 0)

    accepted = bool(p95_ok and no_violations)

    return {
        "accepted": accepted,
        "autoscaling": {
            "p95_ok": p95_ok,
            "no_sla_violations": no_violations,
            "target_utilization_pct": target_util,
            "tolerance_pct": tol,
        },
        "monotonic_ok": None,
        "min_mom_growth_ok": None,
        "private_margin_threshold_pct": targets.get("private_margin_threshold_pct"),
        "capacity_ok": None,
    }
