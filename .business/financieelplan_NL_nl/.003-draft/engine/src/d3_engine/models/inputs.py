"""Input models (scaffold)."""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SimulationPlan:
    pipelines: List[str]
    random_seed: Optional[int] = None
