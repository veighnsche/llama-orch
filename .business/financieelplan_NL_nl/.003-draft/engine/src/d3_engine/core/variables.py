"""Variables grid & treatments.

Responsibilities (spec refs: 16_simulation_variables.md, 20_simulations.md):
- Parse treatments per scope: fixed | low_to_high (grid) | random
- Build cartesian grid over all low_to_high variables (quantized by step)
- Provide iterator of (grid_index, combo) dicts (path -> value)
- (Optional) variable_draws.csv transcript is written by runner.
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
from . import rng


def _frange(min_v: float, max_v: float, step: float) -> List[float]:
    vals: List[float] = []
    if step <= 0:
        return vals
    cur = min_v
    # include max_v with epsilon tolerance
    while cur <= max_v + 1e-12:
        vals.append(round(cur, 12))
        cur += step
    return vals or [min_v]


def build_grid(variables: List[dict]) -> List[Dict[str, float]]:
    """Return a list of parameter combinations (cartesian product of low_to_high numerics).

    variables: CSV-like rows with keys including
      - variable_id, path, type, min, max, step, default, treatment
    Returns combinations mapping variable path to chosen value.
    """
    # Separate numeric low_to_high variables
    grid_axes: List[Tuple[str, List[float]]] = []  # (path, values)
    fixed_defaults: Dict[str, float] = {}
    for row in variables:
        try:
            typ = (row.get("type") or "").strip()
            tr = (row.get("treatment") or "").strip()
            path = (row.get("path") or "").strip()
            if not path:
                continue
            if typ == "numeric":
                if tr == "low_to_high":
                    mn = float(row.get("min") or 0)
                    mx = float(row.get("max") or 0)
                    st = float(row.get("step") or 0)
                    vals = _frange(mn, mx, st)
                    grid_axes.append((path, vals))
                else:  # fixed or others
                    df = float(row.get("default") or 0)
                    fixed_defaults[path] = df
            else:
                # For discrete or non-numeric, default only for now
                df_s = row.get("default")
                if df_s is not None:
                    try:
                        fixed_defaults[path] = float(df_s)
                    except Exception:
                        pass
        except Exception:
            continue

    # Cartesian product
    combos: List[Dict[str, float]] = []
    if not grid_axes:
        combos.append(dict(fixed_defaults))
        return combos

    def _recurse(i: int, cur: Dict[str, float]):
        if i >= len(grid_axes):
            # Fill defaults
            c = dict(fixed_defaults)
            c.update(cur)
            combos.append(c)
            return
        path, vals = grid_axes[i]
        for v in vals:
            cur[path] = v
            _recurse(i + 1, cur)
        cur.pop(path, None)

    _recurse(0, {})
    return combos


def iter_grid_combos(variables: List[dict]) -> Iterable[Tuple[int, Dict[str, float]]]:
    """Yield (grid_index, combo) pairs deterministically."""
    combos = build_grid(variables)
    for i, combo in enumerate(combos):
        yield i, combo


def parse_random_specs(variables: List[dict]) -> List[Tuple[str, float, float]]:
    """Collect numeric random variable specs as (path, min, max)."""
    specs: List[Tuple[str, float, float]] = []
    for row in variables:
        try:
            typ = (row.get("type") or "").strip()
            tr = (row.get("treatment") or "").strip()
            path = (row.get("path") or "").strip()
            if not path or typ != "numeric" or tr != "random":
                continue
            mn = float(row.get("min") or 0)
            mx = float(row.get("max") or 0)
            if mx < mn:
                mn, mx = mx, mn
            specs.append((path, mn, mx))
        except Exception:
            continue
    return specs


def draw_randoms(specs: List[Tuple[str, float, float]], master_seed: int, grid_index: int, replicate_index: int, mc_index: int = 0) -> Dict[str, float]:
    """Draw deterministic random values for given specs using PCG64 substreams.

    Namespacing: ('var_random', path, grid_index, replicate_index, mc_index)
    """
    values: Dict[str, float] = {}
    for path, mn, mx in specs:
        gen = rng.substream(master_seed, "var_random", path, grid_index, replicate_index, mc_index)
        u = float(gen.random())
        values[path] = mn + (mx - mn) * u
    return values
