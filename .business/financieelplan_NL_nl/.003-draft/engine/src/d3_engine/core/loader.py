"""Input loader & overlay (scaffold).
Reads YAML/CSV from inputs/, applies CSV-over-YAML overlay, and yields variable grids.
See .specs/10_inputs.md and .specs/16_simulation_variables.md.
"""

from pathlib import Path
from typing import Iterator, Dict, Any


def load_all(inputs_dir: Path) -> Dict[str, Any]:
    # TODO: implement YAML/CSV reading
    return {"inputs_dir": str(inputs_dir)}


def variable_grid(state: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    # TODO: expand low_to_high into a cartesian product
    yield state
