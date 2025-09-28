"""Aggregation & KPIs (scaffold).

Responsibilities (spec refs: 20_simulations.md §6, 31_engine.md §9):
- Collect job outputs deterministically in (grid_index, replicate_index) order
- Compute percentiles (stochastic.percentiles) and run-level KPIs
- Provide helpers to write consolidated tables via consolidate.py

Implementation to be filled once pipelines return concrete tables.
"""
from __future__ import annotations
from typing import Dict, List, Tuple


def aggregate(job_results: List[Tuple[Tuple[int, int], Dict[str, List[dict]]]]) -> Dict[str, List[dict]]:
    """Aggregate per-job tables into run-level tables (placeholder)."""
    # TODO: implement grouping and percentile computation
    return {}
