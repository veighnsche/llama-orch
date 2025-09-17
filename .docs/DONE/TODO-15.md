# TODO — Full Workspace Review & Retrospective Plan (Spec→Contract→Tests→Code)

This tracker captures the retrospective outcomes from Stage 7 and plans the next cycle across code quality, observability, scheduling, and DX.

## P0 — Blockers (in order)

- [ ] In-file Behavior Tests Initiative (unit tests across crates)
  - Phase 0 — Inventory and prioritization (no code): enumerate critical behavior surfaces to cover with in-file tests and outputs (list + estimates).
    - Targets: `pool-managerd/registry` (health/meta/leases; staleness later), `orchestrator-core/queue` (smoke beyond props), `orchestratord/http/control` (drain/reload/health), `orchestratord/http/data` (admission guardrails), `orchestratord/metrics` (helpers), adapters mock trait conformance sanity.
    - Feasibility: High | Est LOC: 10–20 (planning artifacts only)
  - Phase 1 — Fast wins (low LOC unit tests): implement 1–2 tests per target file listed above.
    - Feasibility: High | Est LOC: 120–180
  - Phase 2 — Policy-sensitive tests (after heartbeat staleness & fairness policy): add staleness → unready test; fairness property un-ignored when hooks land.
    - Feasibility: Medium | Est LOC: 80–120
  - Phase 3 — Maintenance & docs: Testing README for in-file vs BDD scopes, CI wiring, xtask alias to run tests.
    - Feasibility: High | Est LOC: 20–40

  - Discovery Plan — How we will find all in-file tests that need to be written
    1. Source map of candidate files (no code changes):
       - Generate a list of Rust sources to audit (exclude benches/tests/target):
         - Command: `fd -e rs "src" --exclude target --exclude tests --exclude benches -H` (run at repo root)
       - Scope focus: `pool-managerd`, `orchestrator-core`, `orchestratord`, `worker-adapters/*` (mock), and any `plugins/*` with behavior.
       - Per-crate targets (initial sweep):
         - pool-managerd/: `src/registry.rs`, `src/health.rs` (if present), any stateful modules.
         - orchestrator-core/: `src/queue.rs`, `src/lib.rs` exported helpers.
         - orchestratord/: `src/http/control.rs`, `src/http/data.rs`, `src/http/observability.rs`, `src/metrics.rs`, `src/state.rs`, `src/admission.rs`, `src/backpressure.rs`.
         - worker-adapters/*/: public adapter shims; mock adapter behaviors.
         - plugins/: policy host/sdk key entry points.
    2. Static heuristics to identify behaviors likely needing unit tests:
       - Stateful structs and registries (e.g., `Registry`, `QueueWithMetrics`, anything holding `Arc<Mutex<_>>`).
       - Request handlers and boundaries (`pub async fn .. -> Response` in `orchestratord/src/http/*`).
       - Metrics emission points (`metrics::`, `record_*`, `*_TOTAL`, `*_READY`, `*_LEAKS`).
       - Error taxonomy/mapping (`match AdapterErr`, mapping to codes in OpenAPI).
       - Serialization invariants (serde structs with `#[derive(Serialize, Deserialize)]`).
    3. Grep queries to drive the audit:
       - `rg -n "impl\s+.*Registry|QueueWithMetrics|pub\s+async\s+fn\s+.*->\s+Response|metrics::|record_|AdapterErr|serde\(::|derive\(Serialize|Deserialize\)"`
       - For each hit, note: file, behavior, current tests present (yes/no), candidate unit test idea.
       - Optional: list current unit tests to avoid duplication: `rg -n "#\[cfg\(test\)\]|#\[test\]"`.
    4. Cross-reference with BDD coverage (avoid duplication, catch gaps):
       - Map features in `test-harness/bdd/tests/features/**` to the handlers they touch.
       - Identify logic paths only exercised via mocks or side-effects that merit fast unit tests.
    5. Optional coverage-assisted targeting (developer choice):
       - Use coverage tooling to highlight low-covered modules to prioritize.
         - Example: cargo-llvm-cov (Arch/AUR: `cargo-llvm-cov-bin`) or tarpaulin.
         - Run at crate level (e.g., `orchestratord`) and sort by least-covered files.
    6. Produce an inventory artifact for tracking:
       - Create `.docs/testing/infile-test-inventory.md` with columns:
         - `Path | Behavior | Why unit test | Test idea(s) | Status (todo/in-progress/done) | Last updated`
       - Populate from steps 1–4 and update as Phase 1 lands.
    7. Prioritization rules for Phase 1 picks:
       - Critical-path flows (admission/backpressure, drain/reload), state transitions, error mapping, metrics shape, and concurrency safety.
    8. Exit criteria for Phase 0:
       - Every target file has at least one listed behavior and a proposed unit test or a justification for BDD-only.
    9. Deliverables of Phase 0 (no code):
       - `.docs/testing/infile-test-inventory.md` committed with a complete matrix and estimates.
       - An updated TODO with Phase 1 selected test items as subtasks.
    10. Milestones & accountability:
       - M1 (Day 1): Source map and grep findings captured; initial inventory skeleton pushed.
       - M2 (Day 2): Cross-referenced with BDD; prioritized Phase 1 list finalized in TODO.
       - M3 (Day 3–4): Begin Phase 1 implementation in small PR-sized chunks.

    11. Execution checklist (commands + artifacts)
        - [x] Generate source map (repo root):
          - `fd -e rs "src" --exclude target --exclude tests --exclude benches -H > .docs/testing/source-map.txt`
        - [x] Grep behaviors and capture reports:
          - `rg -n "impl\\s+.*Registry|QueueWithMetrics|pub\\s+async\\s+fn\\s+.*->\\s+Response|metrics::|record_|AdapterErr|serde\\(::|derive\\(Serialize|Deserialize\)" --hidden > .docs/testing/grep-map/behaviors.txt`
          - `rg -n "#\\[cfg\\(test\\)\]|#\\[test\]" --hidden > .docs/testing/grep-map/tests-present.txt`
        - [x] Map BDD → handlers:
          - For each feature in `test-harness/bdd/tests/features/**`, note target handler(s) in `.docs/testing/bdd-mapping.md`.
        - [x] Build inventory:
          - Create/maintain `.docs/testing/infile-test-inventory.md` with matrix (Path | Behavior | Why | Idea(s) | Status | Last updated).
        - [ ] Coverage-assisted pass (optional):
          - `cargo llvm-cov --workspace --lcov --output-path coverage.lcov` (or tarpaulin) and list lowest-covered modules per crate.
        - [ ] Update TODO with selected Phase 1 unit tests as subtasks per file.

    12. Acceptance criteria (Phase 0 Definition of Done)
        - [x] Source map, behavior grep, and tests-present reports exist under `.docs/testing/`.
        - [x] `.docs/testing/bdd-mapping.md` links features → handlers.
        - [ ] `.docs/testing/infile-test-inventory.md` is complete for targeted crates and each file has at least one proposed unit test or a justification for BDD-only.
        - [ ] This TODO includes a list of Phase 1 test subtasks with owners and estimates.

    13. Phase 1 candidate tests (initial, to confirm during inventory)
        - pool-managerd
          - [ ] `src/registry.rs`: health/meta get/set happy paths; leases never negative; heartbeat set/get.
        - orchestrator-core
          - [ ] `src/queue.rs`: enqueue/dequeue affects depth; reject vs drop-lru smoke (beyond property tests).
        - orchestratord
          - [x] `src/http/control.rs`: drain(202) → reload(200) toggles readiness; health includes last_error when seeded.
          - [ ] `src/http/data.rs`: guardrails sentinels return expected codes/bodies (INVALID_PARAMS, DEADLINE_UNMET, 429 path).
          - [ ] `src/metrics.rs`: record_stream_started/ended set histogram/gauge/counter labels without panic.
          - [ ] `src/backpressure.rs`: compute_policy_label invariants.
        - worker-adapters/mock
          - [ ] `src/lib.rs`: adapter submit produces token→end sequence in a bounded stream (mocked).

- [ ] Heartbeat Staleness Policy (Spec: OC-POOL-3005)
  - Define threshold for `last_heartbeat_ms` staleness; mark pool Unready when stale.
  - Implement in `/v1/pools/{id}/health` aggregation and add unit test.
  - Feasibility: High | Est LOC: 40–60

- [ ] BDD Coverage: Pool Manager Then Steps
  - Implement assertions for: circuit breaker bounded restarts; device masks; per‑GPU KV cap.
  - Wire minimal hooks/mocks where needed.
  - Feasibility: High | Est LOC: 80–120

- [ ] Scheduling Fairness Hooks (Spec: ORCH-3075..3077)
  - Replace fairness placeholders with scheduler outputs; emit `admission_share`, `deadlines_met_ratio` from policy decisions.
  - Add property test for fairness budget when policy is available.
  - Feasibility: Medium | Est LOC: 120–180

## P1 — Prioritized Next Up

- [ ] Capability Discovery Enhancements
  - Surface `engine_version` and optional `model_digest` in replicasets via features; snapshots/tests.
  - Feasibility: High | Est LOC: 50–80

- [ ] Logging Field Harmonization
  - Use `job_id` consistently; remove redundant `task_id` fields in logs where safe.
  - Feasibility: High | Est LOC: 20–40

- [ ] Pool Health Snapshot Helper
  - Add `state.pool_health_snapshot(pool_id)` to unify reads from registry and in‑memory metrics map.
  - Feasibility: High | Est LOC: 20–30

- [ ] Central Adapter Error Mapping
  - Utility to map `WorkerError` → OpenAPI `ErrorEnvelope`; reuse across handlers.
  - Feasibility: High | Est LOC: 30–45

- [ ] Metrics Label Consistency
  - Ensure `engine_version` labels are consistently present where spec requires; adjust tests.
  - Feasibility: High | Est LOC: 20–30

## Refactoring Opportunities

- [ ] Consolidate Pool State Sources
  - Reduce dual sources (`pool_manager` vs in-memory `pools`) by a unified accessor.

- [ ] Extract Correlation ID Handling
  - Factor common code for `X-Correlation-Id` generation and propagation.

- [ ] Tests Structure
  - Group BDD + provider verify vs unit tests clearly; add a README for test conventions.

## Observability Improvements

- [ ] Structured Logging Redactors
  - Ensure no secrets logged; add redactors and a small compliance test.

- [ ] Metrics Doc Sync
  - Ensure `.specs/metrics/otel-prom.md` includes new gauges/counters added in Stage 7.

## DX/CI Improvements

- [ ] Add `cargo insta review` step docs for snapshots
- [ ] Add a make-like alias or xtask to run the full developer loop

## Progress Log (what changed)

- 2025-09-16 — Initialized Full Workspace Review & Retrospective plan based on Stage 7 completion
  - Added P0/P1 priorities, refactors, observability, and DX tasks.

- 2025-09-17 — Phase 0 discovery artifacts and inline tests
  - Generated grep maps: `.docs/testing/grep-map/behaviors.txt` and `.docs/testing/grep-map/tests-present.txt`.
  - Added BDD mapping: `.docs/testing/bdd-mapping.md`.
  - Created in-file test inventory: `.docs/testing/infile-test-inventory.md`.
  - Implemented inline tests with requirement IDs:
    - `orchestrator-core/src/queue.rs`: OC-CORE-1001, OC-CORE-1002, OC-CORE-1004.
    - `pool-managerd/src/registry.rs`: OC-POOL-3001, OC-POOL-3007.
    - `orchestratord/src/backpressure.rs`: ORCH-2007.
    - `orchestratord/src/metrics.rs`: ORCH-METRICS-0001.
    - `orchestratord/src/http/auth.rs`: ORCH-AUTH-0001.
    - `orchestratord/src/http/control.rs`: ORCH-2006 (drain→reload readiness), ORCH-2003 (health includes last_error).
    - `orchestratord/src/http/data.rs`: ORCH-2001 (INVALID_PARAMS, DEADLINE_UNMET), ORCH-2007 (429 sentinel), OC-CORE-1001/1002 (ADMISSION_REJECT mapping when full).
    - `orchestratord/src/metrics.rs`: ORCH-METRICS-0001 record_stream_started/ended helper smoke tests (labels present in output).
    - `worker-adapters/mock/src/lib.rs`: mock adapter health/props shape and submit stream order (started → token → end).
