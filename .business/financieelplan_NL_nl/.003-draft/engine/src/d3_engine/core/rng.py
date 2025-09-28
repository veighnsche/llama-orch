"""RNG utilities (scaffold)."""
from dataclasses import dataclass


@dataclass
class RNGStream:
    scope: str
    seed: int

    def rand(self) -> float:
        # TODO: replace with numpy Generator(PCG64) substreams
        self.seed = (1103515245 * self.seed + 12345) & 0x7FFFFFFF
        return (self.seed % 100000) / 100000.0
