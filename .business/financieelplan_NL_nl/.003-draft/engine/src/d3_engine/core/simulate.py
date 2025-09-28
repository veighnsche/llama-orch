"""Simulation job executor (scaffold).

Responsibilities (spec refs: 20_simulations.md §3, §3.1; 31_engine.md §8):
- Define a job interface for (grid_index, replicate_index, pipelines, seed)
- Execute public/private pipelines in order and return in-memory tables
- Remain pure/deterministic given inputs; no direct file writes here

Implementation will be completed once pipelines produce concrete table records.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SimJob:
    grid_index: int
    replicate_index: int
    pipelines: List[str]
    seed: Optional[int]


def run_job(job: SimJob, context: Dict) -> Dict[str, List[Dict]]:
    """Run a single simulation job and return a dict of table-name -> rows.

    context: prepared state from loader/validator and variables combo.
    """
    # TODO: Call into pipelines/public/* and pipelines/private/* pure functions
    # Example structure to return once implemented:
    # return {
    #   "public_vendor_choice": [...],
    #   "public_tap_prices_per_model": [...],
    #   "public_tap_customers_by_month": [...],
    #   "public_tap_capacity_plan": [...],
    #   "private_tap_economics": [...],
    #   "private_tap_customers_by_month": [...],
    # }
    return {}
