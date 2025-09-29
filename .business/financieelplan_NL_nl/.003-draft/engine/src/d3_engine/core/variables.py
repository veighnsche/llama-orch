"""Variables grid & treatments.

Responsibilities (spec refs: 16_simulation_variables.md, 20_simulations.md):
- Parse treatments per scope: fixed | low_to_high (grid) | random
- Build cartesian grid over all low_to_high variables (quantized by step)
- Provide iterator of (grid_index, combo) dicts (path -> value)
- (Optional) variable_draws.csv transcript is written by runner.
"""
from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
import itertools
from . import rng


MAX_GRID_POINTS_PER_AXIS = 5  # deterministic cap to avoid combinatorial explosion


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


def _downsample(values: List[float], max_points: int = MAX_GRID_POINTS_PER_AXIS) -> List[float]:
    """Evenly downsample a sorted list to at most max_points, preserving endpoints.

    Deterministic and order-preserving. If len(values) <= max_points, returns as-is.
    """
    n = len(values)
    if n <= max_points or max_points <= 0:
        return values
    if max_points == 1:
        return [values[0]]
    # Compute indices using inclusive linspace
    idxs = [round(i * (n - 1) / (max_points - 1)) for i in range(max_points)]
    # Deduplicate while preserving order
    seen = set()
    ds: List[float] = []
    for i in idxs:
        if i not in seen:
            seen.add(i)
            ds.append(values[int(i)])
    return ds


def _axes_and_defaults(variables: List[dict]) -> Tuple[List[Tuple[str, List[float]]], Dict[str, float]]:
    """Prepare grid axes and defaults with downsampling applied."""
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
                    vals = _downsample(vals, MAX_GRID_POINTS_PER_AXIS)
                    grid_axes.append((path, vals))
                else:
                    df = float(row.get("default") or 0)
                    fixed_defaults[path] = df
            else:
                df_s = row.get("default")
                if df_s is not None:
                    try:
                        fixed_defaults[path] = float(df_s)
                    except Exception:
                        pass
        except Exception:
            continue
    return grid_axes, fixed_defaults


def grid_size(variables: List[dict]) -> int:
    axes, _ = _axes_and_defaults(variables)
    size = 1
    for _, vals in axes:
        size *= max(1, len(vals))
    return size


def iter_grid_combos(variables: List[dict]) -> Iterable[Tuple[int, Dict[str, float]]]:
    """Yield (grid_index, combo) pairs deterministically without full materialization."""
    axes, fixed_defaults = _axes_and_defaults(variables)
    if not axes:
        yield 0, dict(fixed_defaults)
        return
    paths = [p for p, _ in axes]
    values_lists = [vals for _, vals in axes]
    i = 0
    for combo_vals in itertools.product(*values_lists):
        c = dict(fixed_defaults)
        for path, val in zip(paths, combo_vals):
            c[path] = val
        yield i, c
        i += 1


def iter_grid_sampled(variables: List[dict], sample_count: int, master_seed: int) -> Iterable[Tuple[int, Dict[str, float]]]:
    """Yield at most sample_count combinations via deterministic stratified sampling.

    For each axis, we create a deterministic permutation of N buckets and map bucket
    indices to value indices, ensuring broad coverage without full cartesian product.
    """
    axes, fixed_defaults = _axes_and_defaults(variables)
    if not axes:
        yield 0, dict(fixed_defaults)
        return
    N = max(1, int(sample_count))
    paths = [p for p, _ in axes]
    values_lists = [vals for _, vals in axes]
    # Build per-axis permutations deterministically
    perms: List[List[int]] = []
    for path, vals in axes:
        gen = rng.substream(master_seed, "grid_axis_perm", path)
        # Simple Fisher-Yates over 0..N-1
        perm = list(range(N))
        for i in range(N - 1, 0, -1):
            j = int(gen.random() * (i + 1))
            perm[i], perm[j] = perm[j], perm[i]
        perms.append(perm)
    for s in range(N):
        c = dict(fixed_defaults)
        for (vals, perm, path) in zip(values_lists, perms, paths):
            n_i = max(1, len(vals))
            bucket = perm[s]
            idx = min(n_i - 1, int(bucket * n_i / N))
            c[path] = vals[idx]
        yield s, c


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
