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
import numpy as np
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
    # Build per-axis permutations deterministically (vectorized)
    perms: List[List[int]] = []
    for path, _vals in axes:
        gen = rng.substream(master_seed, "grid_axis_perm", path)
        perm = gen.permutation(N).tolist()
        perms.append(perm)
    for s in range(N):
        c = dict(fixed_defaults)
        for (vals, perm, path) in zip(values_lists, perms, paths):
            n_i = max(1, len(vals))
            bucket = perm[s]
            idx = min(n_i - 1, int(bucket * n_i / N))
            c[path] = vals[idx]
        yield s, c


def parse_random_specs(variables: List[dict]) -> List[Dict[str, object]]:
    """Collect numeric random variable specs with distribution metadata.

    Supported treatments:
      - random (alias: random_uniform): uniform[min,max]
      - random_uniform: uniform[min,max]
      - random_normal: normal(mean=default, sd=derived or 10% of default) clamped to [min,max]
      - random_lognormal: lognormal from mean=default, sd=derived (see notes) clamped to [min,max]
      - random_beta: beta with mean=default (percent/fraction), concentration k=50 unless overridden; clamped to [min,max]

    Returns a list of specs dicts containing at least: path, treatment, min, max, default, and optionally aux info.
    """
    # Pre-index rows by path for cross-references (e.g., *_sd)
    by_path: Dict[str, dict] = {}
    for r in variables:
        p = (r.get("path") or "").strip()
        if p:
            by_path[p] = r
    specs: List[Dict[str, object]] = []
    for row in variables:
        try:
            typ = (row.get("type") or "").strip()
            tr = (row.get("treatment") or "").strip().lower()
            path = (row.get("path") or "").strip()
            if not path or typ != "numeric":
                continue
            if tr not in ("random", "random_uniform", "random_normal", "random_lognormal", "random_beta"):
                continue
            mn = float(row.get("min") or 0)
            mx = float(row.get("max") or 0)
            if mx < mn:
                mn, mx = mx, mn
            default = None
            try:
                default = float(row.get("default")) if row.get("default") not in (None, "") else None
            except Exception:
                default = None
            # Heuristic: find a sibling sd for specific known paths if present
            sd_hint = None
            if path.endswith("tokens_per_conversion_mean"):
                # Look for tokens_per_conversion_sd path
                sd_row = by_path.get(path.replace("_mean", "_sd"))
                if sd_row is not None:
                    try:
                        sd_hint = float(sd_row.get("default") or 0.0)
                    except Exception:
                        sd_hint = None
            specs.append({
                "path": path,
                "treatment": ("random_uniform" if tr == "random" else tr),
                "min": mn,
                "max": mx,
                "default": default,
                "sd_hint": sd_hint,
            })
        except Exception:
            continue
    return specs


def draw_randoms(specs: List[Dict[str, object]], master_seed: int, grid_index: int, replicate_index: int, mc_index: int = 0) -> Dict[str, float]:
    """Deterministic per-variable draws using a substream seeded by (seed, path, grid, rep, mc).

    This restores determinism across runs and concurrency regardless of batching.
    """
    import math
    values: Dict[str, float] = {}
    for spec in specs:
        try:
            path = str(spec.get("path"))
            treatment = str(spec.get("treatment") or "random_uniform").lower()
            if treatment == "random":
                treatment = "random_uniform"
            mn = float(spec.get("min") or 0.0)
            mx = float(spec.get("max") or 0.0)
            if mx < mn:
                mn, mx = mx, mn
            default = spec.get("default")
            default_f = float(default) if default is not None else None
            sd_hint = spec.get("sd_hint")
            sd_hint_f = float(sd_hint) if sd_hint is not None else None
            gen = rng.substream(master_seed, "var_random", path, grid_index, replicate_index, mc_index)

            def clamp(x: float) -> float:
                return max(mn, min(mx, x))

            if treatment in ("random_uniform",):
                u = float(gen.random())
                values[path] = mn + (mx - mn) * u
            elif treatment == "random_normal":
                mu = default_f if default_f is not None else (mn + mx) / 2.0
                # sd fallback heuristic
                sd = sd_hint_f if (sd_hint_f not in (None, 0.0)) else ((abs(mu) * 0.1) if mu is not None else (mx - mn) / 6.0)
                draw = float(gen.normal(loc=mu, scale=max(1e-12, sd)))
                values[path] = clamp(draw)
            elif treatment == "random_lognormal":
                m = default_f if default_f is not None else (mn + mx) / 2.0
                s = sd_hint_f if (sd_hint_f not in (None, 0.0)) else ((abs(m) * 0.5) if m is not None else (mx - mn) / 3.0)
                m = max(1e-9, float(m))
                s = max(1e-9, float(s))
                sigma2 = math.log(1.0 + (s * s) / (m * m))
                sigma = math.sqrt(max(1e-12, sigma2))
                mu = math.log(m) - 0.5 * sigma2
                draw = float(gen.lognormal(mean=mu, sigma=sigma))
                values[path] = clamp(draw)
            elif treatment == "random_beta":
                m = default_f
                if m is None:
                    m = (mn + mx) / 2.0 if (mx - mn) > 0 else 0.5
                if m > 1.0:
                    m = m / 100.0
                k = 50.0
                alpha = max(1e-3, m * k)
                beta_param = max(1e-3, (1.0 - m) * k)
                frac = float(gen.beta(alpha, beta_param))
                values[path] = clamp(frac if mx <= 1.0 else frac * 100.0)
            else:
                # Fallback uniform
                u = float(gen.random())
                values[path] = mn + (mx - mn) * u
        except Exception:
            # Midpoint fallback on error
            try:
                mid = (float(spec.get("min") or 0.0) + float(spec.get("max") or 0.0)) / 2.0
                values[str(spec.get("path"))] = mid
            except Exception:
                continue
    return values
