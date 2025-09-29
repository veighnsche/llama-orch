"""Deterministic RNG utilities (PCG64-based).

Responsibilities (spec refs: 33_engine_flow.md, 42_rng_determinism.md):
- Stable seed resolution with precedence: stochastic.random_seed → run.random_seed → operator meta.seed
- Deterministic substreams via stable hashing over (namespace, indices)
- Ready to support per-scope indices: (scope, variable_id, grid_index, replicate_index, mc_index)
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _hash_to_u64(*parts: str) -> int:
    """Hash arbitrary parts to a uint64 suitable for seeding PCG64.

    Uses SHA-256 and truncates to 8 bytes (big-endian) for deterministic seeds.
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return int.from_bytes(h.digest()[:8], "big", signed=False)


@dataclass(frozen=True)
class SeedResolution:
    """Resolve a master seed with explicit precedence.

    Precedence (first non-None wins):
    1) stochastic.random_seed
    2) run.random_seed
    3) operator meta.seed (either public_tap.meta.seed or private_tap.meta.seed)
    """
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


def substream(master_seed: int, namespace: str, *indices: Tuple[object, ...]) -> np.random.Generator:
    """Return a NumPy Generator(PCG64) deterministically derived from master_seed.

    Seed is derived by hashing (master_seed, namespace, indices...).
    """
    parts = [str(master_seed), namespace] + [str(i) for i in indices]
    seed64 = _hash_to_u64(*parts)
    return np.random.Generator(np.random.PCG64(seed64))

def resolve_seed_from_state(state: dict) -> int:
    """Extract and resolve master seed from loaded state with defined precedence.

    Looks at:
    - simulation.stochastic.random_seed
    - simulation.run.random_seed
    - operator.public_tap.meta.seed or operator.private_tap.meta.seed
    Raises ValueError if none present.
    """
    sim = state.get("simulation", {}) if isinstance(state, dict) else {}
    op = state.get("operator", {}) if isinstance(state, dict) else {}
    stoch = sim.get("stochastic", {}) if isinstance(sim, dict) else {}
    run = sim.get("run", {}) if isinstance(sim, dict) else {}
    pub = op.get("public_tap", {}) if isinstance(op, dict) else {}
    prv = op.get("private_tap", {}) if isinstance(op, dict) else {}

    def _meta_seed(d: dict) -> Optional[int]:
        try:
            v = d.get("meta", {}).get("seed")
            return int(v) if v is not None else None
        except Exception:
            return None

    sr = SeedResolution(
        stochastic_seed=(stoch.get("random_seed") if isinstance(stoch, dict) else None),
        run_seed=(run.get("random_seed") if isinstance(run, dict) else None),
        operator_seed=(_meta_seed(pub) or _meta_seed(prv)),
    )
    return sr.resolve()
