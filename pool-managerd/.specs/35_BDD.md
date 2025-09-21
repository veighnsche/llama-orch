# pool-managerd — BDD Delegation (v0)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Delegation

- Cross-crate scenarios owned by root BDD; this crate provides readiness and registry behavior only.

## Boundaries

- BDD drives pool lifecycle through public APIs (HTTP/CLI via orchestrator and daemons) and MUST NOT reach inside this crate.
- GPU-only policy in Home Profile: scenarios assert fail-fast on missing/incompatible GPU; no CPU fallback.

## Feature Catalog (examples)

- Preload & Readiness
  - GIVEN configured pools
  - WHEN preload runs
  - THEN model ensure → engine ensure → health → ready transitions are observed via orchestrator APIs

- Fail-Fast GPU Policy
  - GIVEN no GPU available
  - WHEN preload runs
  - THEN a clear diagnostic is surfaced and pool remains Unready

- Supervision & Backoff (observational)
  - GIVEN a failing engine
  - WHEN supervision restarts occur
  - THEN backoff increases and breaker opens per policy; logs/metrics reflect transitions

## Execution

- Run: `cargo test -p test-harness-bdd -- --nocapture`
- Scope via `LLORCH_BDD_FEATURE_PATH` for pool scenarios

## Traceability

- Aligns with `engine-provisioner` and `model-provisioner` specs; orchestrator capabilities reflect exported pool state

## Refinement Opportunities

- List of step expectations to avoid leaking crate-local state.
