# 48 — Performance & Concurrency

Status: Draft
Version: 0.1.0

## 1. Scope

- Prestatie‑ en gelijktijdigheidsrichtlijnen en tests voor de engine.

## 2. Doelen (SHOULD)

- **E2E minimal fixture**: binnen N seconden (invullen) en RAM < M MB (invullen) op home‑profile hardware.
- **Schaal**: lineaire of sublineaire toename in tijd bij verdubbeling van grid × replicates (indicatief).
- **Concurrency**: `--max-concurrency=N` versnelt zonder nondeterminisme of races.

## 3. Eisen (MUST)

- **Determinisme**: met/zonder parallelisme identieke hashes voor alle artefacten.
- **Threading‑stabiliteit**: BLAS/LAPACK threads gepind; geen nondeterministische kernels.
- **Single‑writer**: centrale schrijver of staging+merge om ordening en bytes te fixeren.

## 4. Tests

- **Smoke**: meet tijd/ram op minimal fixture, log in testoutput.
- **Concurrency**: vergelijk hashes bij `--max-concurrency=1` vs `N`.
- **Stress**: fixture `stress_001/` met p90 drift/FX buffers, autoscaling dicht bij max.

## 5. Implementatie‑richtlijnen

- Gebruik vectorisatie (NumPy/Polars) en vermijd diepe Python loops.
- I/O: kolomformaat en bufferde writes waar mogelijk; schema‑afdwinging in writers.
- Vermijd globale mutable state in pipelines/behavior.

## 6. Referenties

- `21_engine_flow.md`, `41_engine.md`, `40_testing.md`.
