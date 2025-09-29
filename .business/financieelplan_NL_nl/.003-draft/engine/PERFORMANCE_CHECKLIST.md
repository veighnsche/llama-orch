# Performance Checklist (Simulation Hot Path)

This checklist captures high‑risk, high‑reward optimizations focused on the simulation hot path. Code references use current modules and symbols.

## Scope
- Only changes that materially speed up simulation execution time, not one‑time YAML/CSV parsing.
- Primary hot spots:
  - `engine/src/d3_engine/core/variables.py`: `parse_random_specs()`, `iter_grid_sampled()`, `draw_randoms()`
  - `engine/src/d3_engine/aggregate/series.py`: `percentiles_sum_by_month()` and associated series math
  - `engine/src/d3_engine/core/rng.py`: `substream()` seeding and determinism

## Very High‑Risk Wins (During Simulation)

- **[Batch RNG across specs and runs]**
  - What: Vectorize random draws by grouping specs per treatment and drawing arrays in one call.
  - Where: `variables.py::draw_randoms()`
  - Why: Eliminates per‑spec Python overhead; one NumPy call can produce thousands of values.
  - Risks:
    - Determinism changes if substream sequence differs. Current contract: `substream(master_seed, "var_random", path, grid, rep, mc)`.
    - Memory spikes if you generate full replicate×MC slabs.
  - Mitigation:
    - Derive per‑spec seeds via `substream(master, "var_random_batch", spec_index)` then fold in `(grid, rep, mc)`.
    - Unit test: old vs new per‑path values identical for fixed seeds.
  - Suggested steps:
    1) Build packed spec arrays per treatment: `mins[]`, `maxs[]`, `defaults[]`, `sd_hints[]`, `paths[]`.
    2) For each treatment: draw vector with `gen.random(size)`, `gen.normal`, `gen.lognormal`, `gen.beta`.
    3) Clamp vectorized; map back to `path -> value`.

- **[Rust offload (PyO3) for RNG + aggregation]**
  - What: Implement RNG draws and monthly aggregation in Rust; expose FFI to Python.
  - Where: Replace kernels in `variables.py::draw_randoms()` and `series.py::percentiles_sum_by_month()`.
  - Why: Predictable 5–20× speedup on CPU‑bound loops; zero GIL overhead.
  - Risks:
    - FFI build/toolchain complexity; must exactly mirror PCG64 and hash seeding to preserve determinism.
    - Behavior drift (clamping, edge cases) if not mirrored precisely.
  - Mitigation:
    - Use the same SHA‑256 → u64 seed derivation as `rng.py::_hash_to_u64`.
    - Golden tests asserting byte‑for‑byte equality on outputs for fixed seeds.

- **[Streaming approximate percentiles (t‑digest/G–K)]**
  - What: Replace exact sort‑based percentile computation with streaming quantile sketches per month.
  - Where: `aggregate/series.py::percentiles_sum_by_month()`.
  - Why: O(n) streaming vs O(n log n) sorting; large memory reduction across thousands of runs.
  - Risks:
    - Results become approximate; tails may deviate. This is a spec/UX change.
  - Mitigation:
    - Feature flag: `simulation.run.experimental.approx_percentiles: true`.
    - Tolerances in proof bundle: e.g., p50 ±0.25%, p90 ±0.5%.

- **[Columnar spec packing (array model instead of dicts)]**
  - What: Replace dict‑of‑rows with packed arrays for min/max/default/treatment codes.
  - Where: `variables.py::{parse_random_specs,_axes_and_defaults,draw_randoms}`.
  - Why: Removes repeated dict lookups and string processing in inner loops; enables pure array kernels.
  - Risks:
    - Invasive refactor touching CSV load, validator, and any consumer expecting dicts.
  - Mitigation:
    - Keep a translation layer at edges; introduce a packed struct only inside simulation kernels.

- **[Persistent worker processes with pinned state]**
  - What: Run replicates/MC in long‑lived workers that retain parsed inputs, permutations, and RNG state.
  - Where: Orchestration around the engine runner.
  - Why: Avoid repeated setup/deserialization and improve cache locality.
  - Risks:
    - Concurrency bugs, lifecycle leaks; deterministic order must not depend on scheduling.
  - Mitigation:
    - All randomness derived solely from logical indices `(grid, replicate, mc)` so schedule independence holds.

- **[Numba/Cython JIT for numeric kernels]**
  - What: JIT‑compile inner math loops (e.g., fallback math, VAT/cashflow transforms) and feed pre‑drawn random arrays.
  - Where: `variables.py` (if keeping Python RNG), `aggregate/series.py` time‑series ops.
  - Why: 5–10× speedups on tight loops.
  - Risks:
    - Platform/dependency complexity; ensure RNG parity by passing values in, not generating inside JIT.

- **[Cache permutations/specs across runs]**
  - What: Cache results of `iter_grid_sampled()` permutations and `parse_random_specs()` across invocations keyed by input hash.
  - Where: `variables.py::{iter_grid_sampled, parse_random_specs}`.
  - Why: Eliminates repeated recomputation across pipelines or reruns.
  - Risks:
    - Stale cache if inputs change; requires robust invalidation (hash of CSV contents + parameters).

## Determinism Invariants (must hold)
- `rng.substream(master, ns, path, grid, rep, mc)` yields identical sequences for identical arguments regardless of scheduling.
- `draw_randoms()` emits identical values per `path` for the same `(master_seed, grid_index, replicate_index, mc_index)`.
- Any approximate percentile mode must be opt‑in and documented with error bounds.

## Rollout & Validation Plan
- **Feature flags** (suggested keys in `simulation.yaml`):
  - `run.experimental.vectorized_rng: true|false`
  - `run.experimental.approx_percentiles: true|false`
  - `run.experimental.rust_kernels: true|false`
- **Golden tests**:
  - Fixed seed, small grid; compare JSON/CSV outputs old vs new.
  - Statistical sanity: distributions preserve mean/variance after batching.
- **Bench harness**:
  - Measure wall‑clock for 1, 10, 50 replicates × 25 MC with grid=1.
  - Track CPU time and memory with `cProfile` and RSS metrics.

## Quick Profiling Commands
```bash
# Whole run profile (from engine/ entrypoint)
python -m cProfile -o profile.out main.py
python - <<'PY'
import pstats; s=pstats.Stats('profile.out'); s.sort_stats('cumtime').print_stats(30)
PY
```

## Current low‑risk improvements already applied
- `variables.py` now uses `gen.normal`, `gen.lognormal`, `gen.beta`, and `gen.permutation()`.
- `series.py` sorts once per month and reuses the sorted list for multiple percentiles.
