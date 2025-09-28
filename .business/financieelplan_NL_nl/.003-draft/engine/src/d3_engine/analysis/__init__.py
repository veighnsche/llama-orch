"""Analysis layer (KPIs, percentiles, sensitivity) — scaffold.

Purpose:
- Encapsulate data analysis on simulation outputs (after per-job results are available)
- Compute KPIs, percentiles and sensitivity across grid/replicates/MC
- Keep separate from I/O (writers) and from core orchestration

Public entrypoint:
- analyze(tables: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]
"""
from __future__ import annotations
from typing import Any, Dict

from . import kpis, percentiles, sensitivity


def analyze(tables: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Run analysis over tables. Placeholder implementation.

    Returns a dict with optional sections, e.g., {"kpis": {...}, "sensitivity": {...}}
    """
    # Wire percentiles to read outputs from out_dir; kpis/sensitivity remain placeholders
    return {
        "kpis": kpis.compute_kpis(tables, context),
        "percentiles": percentiles.compute_percentiles(tables, context),
        "sensitivity": sensitivity.compute_sensitivity(tables, context),
    }
