"""Output models (scaffold)."""
from dataclasses import dataclass
from typing import List


@dataclass
class ArtifactIndex:
    files: List[str]
