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
    """Placeholder acceptance. Implement proper checks when tables are populated."""
    return {
        "accepted": True,
        "monotonic_ok": None,
        "min_mom_growth_ok": None,
        "private_margin_threshold_pct": targets.get("private_margin_threshold_pct"),
        "capacity_ok": None,
    }
