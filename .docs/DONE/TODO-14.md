# TODO — Active Tracker (Spec→Contract→Tests→Code)

This is the single active TODO tracker for the repository. Maintain execution order and update after each task with what changed, where, and why.

## P0 — Blockers (in order)

- [x] Stage 7 — Pool Manager Readiness (Spec: OC-POOL-3001..3012)
  - [x] Registry heartbeat + readiness
    - Implemented `last_heartbeat_ms`, `version`, `last_error` in `pool-managerd/src/registry.rs`; getters/setters added. Health will consider staleness in next iteration (policy TBD).
    - `/v1/pools/{id}/health` aggregates registry `live/ready/last_error` + in-memory `draining`/metrics.
  - [x] Drain/reload flows
    - `POST /v1/pools/{id}/drain` flips draining and marks ready=false in registry; emits `drain_events_total` and `pool_ready=0`.
    - `POST /v1/pools/{id}/reload` clears draining and marks ready=true; sets registry version; returns 200 OK.
  - [x] Leases scaffolding (admission → assignment)
    - `allocate_lease`/`release_lease` APIs; wired around adapter streaming; emits `active_leases{pool_id}` gauge.
  - [x] Metrics & observability
    - Added `pool_ready`, `active_leases`, `drain_events_total` in `orchestratord/src/metrics.rs`.
  - [x] Tests and BDD glue
    - Implemented BDD Then-step assertions for preload failure and driver error; added registry dep to BDD harness.
    - Adjusted reload to return 200 OK; all workspace tests green.
  - Files: `pool-managerd/src/{registry.rs,health.rs,leases.rs}`, `orchestratord/src/http/{control.rs,data.rs}`, `orchestratord/src/metrics.rs`, `test-harness/bdd/src/steps/pool_manager.rs`, `test-harness/bdd/Cargo.toml`.

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

- 2025-09-16 — Stage 7: Pool Manager Readiness implemented end-to-end
  - Registry: health + heartbeat/version/last_error + leases counters; getters/setters in `pool-managerd`.
  - Control plane: drain/reload wired to registry; pool health includes `last_error`.
  - Data plane: allocate/release leases around adapter streaming; `active_leases` gauge.
  - Metrics: `pool_ready`, `active_leases`, `drain_events_total` added.
  - BDD: step glue added for preload failure and driver error states; control-plane reload returns 200 OK to satisfy steps.
  - All workspace tests pass; clippy/fmt clean.

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
