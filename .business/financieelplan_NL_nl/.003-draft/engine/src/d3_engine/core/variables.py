"""Variables grid & treatments (scaffold).

Responsibilities (spec refs: 16_simulation_variables.md, 20_simulations.md):
- Parse treatments per scope: fixed | low_to_high (grid) | random
- Build cartesian grid over all low_to_high variables (quantized by step)
- Provide iterator of (grid_index, combo) dicts per scope
- (Optional) Emit variable_draws.csv transcript for random draws per replicate

Implementation to be completed when loader/validator are ready.
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple


def build_grid(variables: List[dict]) -> List[Dict[str, float]]:
    """Return a list of parameter combinations (placeholder).

    variables: rows shaped like VariableRow but kept generic here to avoid tight coupling.
    """
    # TODO: implement proper grid expansion from min/max/step for numeric low_to_high
    # Placeholder: single empty combination
    return [{}]


def iter_grid_combos(variables: List[dict]) -> Iterable[Tuple[int, Dict[str, float]]]:
    """Yield (grid_index, combo) pairs deterministically."""
    combos = build_grid(variables)
    for i, combo in enumerate(combos):
        yield i, combo
