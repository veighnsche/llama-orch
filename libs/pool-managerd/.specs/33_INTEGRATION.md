# pool-managerd — Integration Tests (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Scope

- Preload pipeline (model ensure → engine ensure → health → ready).

## Test Catalog

- Preload Sequencing
  - GIVEN empty registry
  - WHEN preload is triggered for pool X
  - THEN operations occur in order: model ensure → engine ensure → health probe → ready

- GPU-Only Fail-Fast
  - GIVEN no GPU or incompatible driver
  - WHEN preload runs
  - THEN pool enters `Failed(GpuRequired)` (or equivalent) with clear diagnostics; no CPU fallback attempted

- Concurrency & Throttling
  - GIVEN multiple pools/replicas
  - WHEN preload runs
  - THEN no more than N concurrent operations occur per config; others queue deterministically

- Restart & Backoff
  - GIVEN a failing health probe
  - WHEN supervision restarts
  - THEN backoff grows to configured cap; jitter (if enabled) remains within bounds

- Capability Export
  - WHEN ready
  - THEN exported snapshot to orchestrator-core includes accurate `slots`, `ctx_max`, `max_tokens_out`, `engine_version`, `sampler_profile_version`

## Fixtures & Mocks

- Stub provisioners (model/engine) that can succeed/fail deterministically with injected errors
- Fake health checker with controllable outcomes and latencies
- Deterministic clock for backoff assertions

## Execution

- `cargo test -p pool-managerd -- --nocapture`

## Traceability

- Home profile GPU-only policy (fail-fast)
- Aligns with `provisioners/*` specs and `orchestrator-core` snapshot expectations

## Refinement Opportunities

- Supervisory restart storm simulations with deterministic clocks.
