# TODO — Active Tracker (Spec→Contract→Tests→Code)

This is the single active TODO tracker for the repository. Maintain execution order and update after each task with what changed, where, and why.

## P0 — Blockers (in order)

- [ ] Stage 7 — Pool Manager Readiness (Spec: OC-POOL-3001..3012)
  - [ ] Registry heartbeat + readiness [Feasibility: High | Est LOC: 60–90]
    - Add `last_heartbeat_ms`, `ready`, `version` fields to `pool-managerd` registry entries; getters/setters (thread-safe). [30–45 LOC]
    - Wire `/v1/pools/{id}/health` aggregation (live/ready/draining/metrics) from registry. [20–30 LOC]
    - Add sanity unit tests for set/get and stale heartbeat -> unready. [10–15 LOC]
  - [ ] Drain/reload flows [Feasibility: Medium | Est LOC: 70–110]
    - Update handlers to call registry: `drain(pool_id, until)` and `reload(pool_id, new_model_ref)`. [30–40 LOC]
    - Structured logs and metrics for lifecycle, include `pool_id`, `engine`, `engine_version`. [20–30 LOC]
    - Rollback placeholders (no-op ok for v-slice) and deadline handling. [20–40 LOC]
  - [ ] Leases scaffolding (admission → assignment) [Feasibility: Medium | Est LOC: 80–120]
    - Introduce `LeaseId` allocation/return; track `active_leases{pool_id}` gauge. [50–70 LOC]
    - Invariants: non-negative active leases; simple counter-based allocator. [15–25 LOC]
    - Hook admission path to increment/decrement around dispatch (mocked). [15–25 LOC]
  - [ ] Propagate `engine_version` and `model_digest` [Feasibility: High | Est LOC: 35–60]
    - Surface adapter `engine_version()` in replicasets; add optional `features.engine_version`. [20–35 LOC]
    - Optional `model_digest` plumbed via features map (no OpenAPI change). [15–25 LOC]
  - [ ] Metrics & observability [Feasibility: High | Est LOC: 50–80]
    - Gauges: `pool_ready{pool_id}`, `active_leases{pool_id}`; Counter: `drain_events_total{pool_id,reason}`. [30–45 LOC]
    - Linter parity and label budgets; ensure `engine_version` label used where applicable. [20–35 LOC]
  - [ ] Tests (prove requirements) [Feasibility: High | Est LOC: 60–90]
    - Unit: registry heartbeat/ready; leases counters. [20–30 LOC]
    - BDD: pool health transitions (preload failure, driver error backoff) glue with assertions. [20–40 LOC]
    - Snapshots: replicasets include slot counts; stable ordering. [20–30 LOC]
  - Files: `pool-managerd/src/{registry.rs,health.rs,leases.rs}`, `orchestratord/src/http/control.rs`, `orchestratord/src/state.rs`, tests under `orchestratord/tests/`, `cli/consumer-tests/tests/`.

## BDD Glue Audit — Coverage and Gaps

- Pool Manager lifecycle feature: `test-harness/bdd/tests/features/pool_manager/pool_manager_lifecycle.feature`
  - Step glue present in `test-harness/bdd/src/steps/pool_manager.rs`, but several Then steps are placeholders with no assertions:
    - `then_pool_readiness_false_last_error_present()` — empty body; should assert readiness=false and an error cause present via world state or API call.
    - `then_pool_unready_and_restarts_with_backoff()` — empty; should assert transitions with a backoff schedule.
    - `then_restart_storms_bounded_by_circuit_breaker()` — empty; add assertions for bounded restarts.
    - `then_placement_respects_device_masks_no_spill()` — empty; should validate placement respects device masks.
    - `then_per_gpu_kv_capped_smallest_gpu()` — empty; add assertion scaffold.
  - ACTION (P0 under Stage 7 Tests): Implement assertions in these steps and wire to orchestrator state/registry or mocks. [Feasibility: High | Est LOC: 60–90]

- Scheduling features (representative files under `test-harness/bdd/tests/features/scheduling/`)
  - Glue appears present in `steps/scheduling.rs` and `steps/deadlines_preemption.rs`, but some are placeholders; verify per-scenario upon Stage 8.

- Data/Control plane glue broadly covered in `steps/{data_plane,control_plane,catalog}.rs` and match features; no immediate gaps blocking Stage 7.

## Refactoring Opportunities

- Logging field harmonization: adopt `job_id` consistently and de-duplicate `task_id` vs `job_id` in `orchestratord/src/http/data.rs`. [Feasibility: High | Est LOC: 10–20]
- Adapter error mapping: extract a small utility to convert `WorkerError` -> `ErrorEnvelope` (OpenAPI), used across handlers. [Feasibility: High | Est LOC: 20–30]
- Pool health wiring: factor a small helper `state.pool_health_snapshot(pool_id)` to unify reads from `pool_manager` and `pools` metrics map. [Feasibility: High | Est LOC: 20–30]
- Metrics labels: ensure `engine_version` included consistently; centralize via wrapper functions in `orchestratord/src/metrics.rs`. [Feasibility: High | Est LOC: 20–30]

## Small Bugs Fixed (<30 LOC)

- Clippy and style fixes: removed unnecessary casts and lifetimes; simplified guard in `http/auth.rs`; replaced unit-struct `.default()`; removed unused import; added `walkdir` dep for gating helper. Total ~25 LOC across files.

## P1 — Next up (ordered)

- [ ] Policy integration hooks for scheduling fairness (ORCH-3075..3077)
  - Emit real `admission_share` and `deadlines_met_ratio` from scheduler decisions.
- [ ] Capability discovery: add `engine_version` exposure strategy (contract note) and snapshots.
- [ ] Harden structured logging and redactors; ensure zero secret leakage.

## Progress Log (what changed)

- 2025-09-16 — Initialized Stage 7 plan (Pool Manager Readiness) per README_LLM.md
  - Added execution plan covering registry readiness, drain/reload, leases, metrics, tests, and observability.

---

## Daily Developer Loop (run locally before pushing)

- [ ] `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings`
- [ ] `cargo xtask regen-openapi && cargo xtask regen-schema`
- [ ] `cargo run -p tools-spec-extract --quiet && git diff --exit-code`
- [ ] `cargo test --workspace --all-features -- --nocapture`
- [ ] `bash ci/scripts/check_links.sh`

---

## PR Checklist (run before merge)

- [ ] Specs/Contracts
  - [ ] Spec updated or explicitly confirmed unchanged; requirement IDs present
  - [ ] OpenAPI/config schema updated when surfaces change; examples compile
- [ ] Proofs
  - [ ] Requirements regen is clean: `cargo run -p tools-spec-extract --quiet && git diff --exit-code`
  - [ ] Links are valid: `bash ci/scripts/check_links.sh`
  - [ ] Workspace healthy: `cargo fmt` + `clippy -D warnings`
  - [ ] Tests pass: `cargo test --workspace --all-features -- --nocapture`
- [ ] Hygiene
  - [ ] No BC shims or dead code left behind
  - [ ] Commits and PR description reference relevant requirement IDs
  - [ ] TODO tracker updated with what changed; archive via `ci/scripts/archive_todo.sh` when complete
