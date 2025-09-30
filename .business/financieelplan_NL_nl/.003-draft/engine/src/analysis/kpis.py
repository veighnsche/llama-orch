"""KPIs computation (scaffold)."""
from __future__ import annotations
from typing import Any, Dict


def compute_kpis(tables: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Compute high-level KPIs from simulation tables and context.

    Currently passes through autoscaling summary as KPIs.autoscaling.
    """
    out: Dict[str, Any] = {}
    autoscaling_summary = context.get("autoscaling_summary")
    if autoscaling_summary:
        out["autoscaling"] = autoscaling_summary
    return out
