"""Deterministic RNG (placeholder for PCG64).

Provides seed resolution and substream derivation via stable hashing.
"""
from __future__ import annotations
import hashlib
import random
from dataclasses import dataclass
from typing import Optional, Tuple


def _hash_to_int(*parts: str) -> int:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    # Take 8 bytes to fit into Python int reasonably
    return int.from_bytes(h.digest()[:8], "big", signed=False)


@dataclass(frozen=True)
class SeedResolution:
    stochastic_seed: Optional[int]
    run_seed: Optional[int]
    operator_seed: Optional[int]

    def resolve(self) -> int:
        if self.stochastic_seed is not None:
            return int(self.stochastic_seed)
        if self.run_seed is not None:
            return int(self.run_seed)
        if self.operator_seed is not None:
            return int(self.operator_seed)
        raise ValueError("No seed provided (stochastic/run/operator)")


def substream(master_seed: int, namespace: str, *indices: Tuple[object, ...]) -> random.Random:
    """Return a Random instance deterministically derived from a master seed.

    This mimics PCG64 substreams by hashing master_seed + namespace + indices.
    """
    parts = [str(master_seed), namespace] + [str(i) for i in indices]
    seed = _hash_to_int(*parts)
    rng = random.Random(seed)
    return rng

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
