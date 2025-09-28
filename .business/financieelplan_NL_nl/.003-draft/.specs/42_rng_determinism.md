# 42 — RNG & Determinisme

Status: Draft
Version: 0.1.0

## 1. Scope

- Seed‑hiërarchie, RNG‑substreams en determinisme over grid/replicates/MC en parallelisme.
- Transcript (`variable_draws.csv`) en stabiliteit van numerieke paden.

## 2. Eisen (MUST)

- **Seed‑resolutie**: `stochastic.random_seed` → `run.random_seed` → `operator/<tap>.yaml: meta.seed` → anders **ERROR`.
- **Substreams**: PCG64 met hashing `H(master_seed, namespace, scope, variable_id?, grid_index, replicate_index, mc_index?)`.
- **Step‑kwantisatie**: `min + k*step` binnen `[min,max]`.
- **Concurrency‑stabiliteit**: `run.max_concurrency` beïnvloedt performance, niet output bytes/hashes.
- **Transcript**: `variable_draws.csv` SHOULD aanwezig zijn voor random variabelen.
- **BLAS‑threads**: pinnen (bijv. `OPENBLAS_NUM_THREADS=1`) voor stabiele floats (SHOULD).

## 3. Tests

- **Determinism**: twee runs met identieke seeds → `SHA256SUMS` gelijk; `determinism.feature`.
- **Concurrency**: runs met `--max-concurrency=1` vs `=N` → identieke hashes.
- **Grid‑randen**: values precies op `min/max` na kwantisatie; geen off‑by‑one in `step`.
- **Discrete**: uniforme trekking uit `notes.values`; transcript weerspiegelt keuzes.

## 4. Artefacten

- `variable_draws.csv` met kolommen: `scope,variable_id,path,grid_index,replicate_index,draw_value`.

## 5. Referenties

- `21_engine_flow.md`, `16_simulation_variables.md`, `41_engine.md`.
