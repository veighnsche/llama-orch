# TODO — Active Tracker (Spec→Contract→Tests→Code)

This is the single active TODO tracker for the repository. Maintain execution order and update after each task with what changed, where, and why.

## P0 — Blockers (in order)

- [ ] Stage 6 vertical slice (Admission → Dispatch → SSE) — Spec: ORCH-2001/2002/2003, OC-CTRL-2010/2012, OC-CORE-1001..1005
  - [x] Implement enqueue into `QueueWithMetrics`
  - [x] Minimal placement hook and dispatch via a `WorkerAdapter`
  - [x] Map adapter stream to contract SSE events (`started|token|metrics|end`) with budget headers
  - [x] Metrics side-effects for first token and end
  - Files: `orchestratord/src/http/data.rs`, `orchestratord/src/state.rs`, `orchestratord/src/placement.rs`, `worker-adapters/*/`.

- [x] Fix SSE token event shape to match OpenAPI (`SSEToken { t, i }`) — Spec: OC-CTRL-2020/2022; OpenAPI: `components/schemas/SSEToken`
  - Replaced `{"text":"hello"}` with `{ "t": "...", "i": N }` in `stream_task()` and adapter stubs.
  - Files: `orchestratord/src/http/data.rs`, `worker-adapters/*/src/lib.rs`.

- [x] Add `/v1/capabilities` route and handler — Spec: OC-CTRL-2060/2061; OpenAPI: `contracts/openapi/control.yaml`
  - Implemented handler returning `api_version`, per-engine `ctx_max`, `supported_workloads`, `features`, `rate_limits`.
  - Files: `orchestratord/src/http/control.rs`, `orchestratord/src/lib.rs` (route wiring).

- [x] Align Catalog/Lifecycle routes with OpenAPI paths — Spec: ORCH-3060..3074; OpenAPI: `contracts/openapi/control.yaml`
  - Changed routes to `/v1/catalog/models`, `/v1/catalog/models/{id}`, `/v1/catalog/models/{id}/verify`, `/v1/catalog/models/{id}/state`.
  - Files: `orchestratord/src/lib.rs`, `orchestratord/src/http/catalog.rs`, `orchestratord/src/http/control.rs`.

- [x] Regenerator drift: update generated data-plane types for new error kinds — Spec: ORCH-2006/2007, v3.2 addenda
  - Ensured `contracts/api-types/src/generated.rs` reflects `ErrorKind` union including `DEADLINE_UNMET`, `MODEL_DEPRECATED`, `UNTRUSTED_ARTIFACT` and advisory fields.
  - Updated template `xtask/src/templates/generated_api_types.rs`; verified `cargo xtask regen-openapi` is idempotent.

- [x] Correlation ID handling standardization across responses/SSE — Proposal: 2025-09-15-ux-dx-improvements.md
  - Echoed `X-Correlation-Id` on responses and SSE; added optional budget headers.
  - Files: `orchestratord/src/http/{data.rs,control.rs,catalog.rs}`.

- [ ] Observability: structured logs with required fields — Spec: ORCH-3027/3028; `.specs/metrics/otel-prom.md`
  - Replace in-memory `Vec<String>` logs with `tracing` JSON logs. Include required fields: `job_id`, `session_id`, `engine`, `pool_id`, `replica_id`, `model_id`, `quant`, `ctx`, `kv_warmth`, timings, etc.
  - Files: `orchestratord/src/http/handlers.rs`, introduce a logging layer; ensure no secrets/API keys in logs.

- [ ] Metrics parity and labels — Spec: `.specs/metrics/otel-prom.md`; Linter: `ci/metrics.lint.json`
  - Double-check label sets in `orchestratord/src/metrics.rs` match linter, esp. exception on `tasks_rejected_total` (omit `engine_version`).
  - Add emission sites for fairness and deadlines gauges when wired (Stage 9).

- [x] Pool Manager readiness scaffold (minimum to unblock Stage 6/7) — Spec: OC-POOL-3001..3012
  - Provided basic registry/health/readiness state and wired `/v1/pools/{id}/health` to real state.
  - Files: `orchestratord/src/state.rs`, `orchestratord/src/http/control.rs`.

## P1 — Next up (ordered)

- [ ] Scheduling & fairness wiring (WFQ, quotas, session affinity) — Spec: ORCH-3075..3077; OC-CORE fairness; export `admission_share`/`deadlines_met_ratio`.
- [ ] Capability discovery snapshots and provider verify — implement GET `/v1/replicasets` payload enrichment (if not using `/v1/capabilities`).
- [ ] Policy host + SDK minimal assertions and CDC tests — Specs: OC-POLICY-4001..4011, OC-POLICY-SDK-4101..4112.
- [ ] Startup self-tests (preload, minimal decode, cancel, telemetry) — Spec: ORCH-3049.
- [ ] Real-model E2E (Haiku) on a live worker (GPU preferred; CPU CI fallback) — Workflow §4.

## Progress Log (what changed)

- 2025-09-16 — Repository audit + TODO refresh per `README_LLM.md` discipline
  - Findings: code/spec alignment mostly good for scaffolding, but several release blockers identified.
    - SSE token event shape in `orchestratord/src/http/handlers.rs` uses `{ "text": "hello" }`; must be `{ "t": "...", "i": N }` (OC-CTRL-2020/2022; `contracts/openapi/data.yaml`).
    - Missing `/v1/capabilities` route/handler; add per OC-CTRL-2060/2061.
    - Catalog routes differ from OpenAPI (`/v1/catalog/models`…); align server wiring.
    - `contracts/api-types/src/generated.rs` `ErrorKind` lacks v3.2 additions (`DEADLINE_UNMET`, `MODEL_DEPRECATED`, `UNTRUSTED_ARTIFACT`); update `xtask` templates and regen.
    - Logs are in-memory strings; replace with `tracing` JSON logs including ORCH-3027 fields; ensure no secret leakage.
    - Metrics vectors present and close to spec; verify linter parity and extend emissions where needed (fairness/deadlines).
  - Scope scanned:
    - Specs: `.specs/00_llama-orch.md`, `.specs/10-orchestrator-core.md`, `.specs/20-orchestratord.md`, proposals v3.2; metrics spec.
    - Contracts: `contracts/openapi/{data.yaml,control.yaml}`, config schema lib/tests.
    - Code: `orchestrator-core` queue + properties; `orchestratord` handlers/state/metrics; adapter stubs; pool-managerd stubs; `xtask` regenerators; CI pipelines and linter.
  - Next action: tackle P0 items in order starting with Stage 6 vertical and SSE/event shape fixes; update this tracker after each PR.

- 2025-09-16 — Modularize HTTP and complete P0 contract alignments
  - Modularized `orchestratord/src/http/handlers.rs` into:
    - `orchestratord/src/http/auth.rs`, `data.rs`, `catalog.rs`, `control.rs`, `observability.rs`; routes updated in `orchestratord/src/lib.rs`.
  - Completed P0s:
    - SSE token shape `{t,i}` and SSE end payload — `http/data.rs`, adapters updated in `worker-adapters/*/src/lib.rs`.
    - `/v1/capabilities` — `http/control.rs`, route wired in `lib.rs`.
    - Catalog/lifecycle path alignment — routes in `lib.rs`; handlers in `http/{catalog.rs,control.rs}`.
    - Regenerator drift — extended `xtask/src/templates/generated_api_types.rs`; ran `cargo xtask regen-openapi` to update `contracts/api-types/src/generated.rs`.
    - Correlation ID and budget headers — added across `http/{data,control,catalog}.rs` and SSE.
  - Pool health: added minimal registry in `orchestratord/src/state.rs`; wired `GET /v1/pools/{id}/health` in `http/control.rs`.
  - Observability: added `tracing` logs for admission and lifecycle; temporarily retained in-memory logs to keep BDD steps green.
  - Stage 6 progress (vertical slice):
    - Added `orchestratord/src/placement.rs` with a minimal adapter chooser by engine.
    - Extended `AppState` with `adapters` registry (mock) and `sse` transcript store.
    - `create_task()` now spawns a background dispatch that builds an SSE transcript from adapter events (started → token → metrics → end), stored in-memory and served by `stream_task()`.
    - Added metrics side-effects for first token latency and end decode latency using `metrics::record_stream_started/ended`.
    - Adjusted BDD harness to avoid header trait mismatch and invalid Debug formatting; all workspace tests pass.

- 2025-09-16 — Week 3 & 4: pool readiness, structured logs, adapters, capabilities & fairness placeholders
  - Pool readiness: added `pool-managerd` registry with health getters/setters; orchestrator `/v1/pools/{id}/health` now backed by registry state. (Spec: OC-POOL-3001..3012)
  - Structured logs: enabled JSON logs; admission/stream logs include required fields (`job_id`, `session_id`, `engine`, `pool_id`, `replica_id`, `ctx`, timings). (Spec: ORCH-3027/3028)
  - Worker adapters: added typed `WorkerError`; mapped adapter errors to SSE `error` frames with contract `code/message/engine`. (Spec: OC-ADAPTER-5001..5006)
  - Capabilities: added insta snapshots for capability discovery and enriched `/v1/replicasets` payload using adapter props. (Spec: OC-CTRL-2060/2061)
  - Scheduling prep: wired fairness gauges placeholders (`admission_share`, `deadlines_met_ratio`) emission; fairness property remains ignored until policy finalization. (Spec: ORCH-3075..3077)
  - Haiku E2E gating: helpers for `REQUIRE_REAL_LLAMA=1` and anti-cheat scan; no test enables real runs yet.

- 2025-09-16 — Stage 6: enqueue path and 429 mapping
  - Wired HTTP admission to `QueueWithMetrics::enqueue(...)`; map `QueueFullReject` to 429 with `Retry-After` and `X-Backoff-Ms` headers and advisory fields (`policy_label`, `retriable`, `retry_after_ms`).
  - Metrics: increment `tasks_enqueued_total`, update `queue_depth`; on reject record `admission_backpressure_events_total` and `tasks_rejected_total`.
  - Files: `orchestratord/src/http/data.rs`.
  - Spec: ORCH-2001 (admission), ORCH-2007 (429 backpressure), OC-CORE-1001..1002 (queue invariants).

---

## Roadmap — 4 Weeks Plan (Spec→Contract→Tests→Code)

Discipline: follow `README_LLM.md` strictly. Each item should link back to spec IDs and adjust OpenAPI/config schema before code when surfaces change.

### Week 1 — Contract alignment + small deltas

- [x] Fix SSE token shape `{t, i}` in `stream_task()` and adapters — Spec: OC-CTRL-2020/2022
  - Code: `orchestratord/src/http/data.rs`, `worker-adapters/*/src/lib.rs`
- [x] Add `/v1/capabilities` GET — Spec: OC-CTRL-2060/2061
  - Code: `orchestratord/src/http/control.rs` (new handler), `orchestratord/src/lib.rs` (route)
- [x] Align catalog/lifecycle paths with OpenAPI
  - Code: `orchestratord/src/lib.rs` routes; `orchestratord/src/http/{catalog.rs,control.rs}`
- [x] Standardize `X-Correlation-Id` across responses and SSE
  - Code: `orchestratord/src/http/{data.rs,control.rs,catalog.rs}`
- [x] Update OpenAPI‑generated types and templates; run deterministic regen
  - Code: `xtask/src/templates/generated_api_types.rs`; ran `cargo xtask regen-openapi && cargo xtask regen-schema`

### Week 2 — Stage 6 vertical: Admission → Dispatch → SSE

- [x] Minimal placement and dispatch path; wire adapter `submit()` from handler
  - Code: `orchestratord/src/state.rs`, `orchestratord/src/placement.rs`, adapters
- [x] SSE mapping and backpressure headers on stream (mapping complete; backpressure headers present on SSE start)
  - Code: `orchestratord/src/http/data.rs`, `orchestratord/src/backpressure.rs`
- [x] Metrics side‑effects on start/end (first token latency, decode latency)
  - Code: `orchestratord/src/metrics.rs`, `orchestratord/src/http/data.rs`
- [x] Provider verify remains green; tests cover SSE contract
  - Code: `orchestratord/tests/provider_verify.rs`
  - Note: Enqueue into `QueueWithMetrics` remains TODO to fully close Stage 6.

### Week 3 — Pool readiness + observability

- [x] Pool manager minimal readiness backing `/v1/pools/{id}/health`
  - Code: `pool-managerd/src/registry.rs` (new in-memory registry APIs), `orchestratord/src/state.rs` (registry in `AppState`), `orchestratord/src/http/control.rs` (handler reads registry). Spec: OC-POOL-3001..3012
- [x] Structured logs via `tracing` with required fields; redact secrets
  - Code: `orchestratord/src/main.rs` (JSON logs), `orchestratord/src/http/data.rs` (admission/stream logs with job_id, session_id, engine, pool_id, replica_id, timings); ensured no secrets/API keys in logs. Spec: ORCH-3027/3028
- [x] Worker adapters: mock + llamacpp conformance basics; typed error mapping
  - Code: `worker-adapters/adapter-api` adds typed `WorkerError`; `orchestratord/src/http/data.rs` maps errors to SSE `error` frames. Spec: OC-ADAPTER-5001..5006

### Week 4 — Capabilities verification + scheduling prep

- [x] Capability discovery snapshots and provider verify
  - Code: `cli/consumer-tests/tests/snapshot_capabilities.rs` (insta snapshots for capabilities and replicasets); provider verify already covers `GET /v1/capabilities`. Spec: OC-CTRL-2060/2061
- [x] Scheduling placeholders: fairness gauges emission sites; unignore fairness property when ready
  - Code: `orchestratord/src/http/data.rs` emits `admission_share` and `deadlines_met_ratio` placeholders after first token. Fairness property test remains `#[ignore]` until policy finalized.
- [x] Prep E2E Haiku environment; ensure gating `REQUIRE_REAL_LLAMA=1`
  - Code: `test-harness/e2e-haiku/src/lib.rs` adds `require_real_llama_env()` and `anti_cheat_scan_repo()`; added `walkdir` dep.

---

## Daily Developer Loop (run locally before pushing)

- [ ] `cargo fmt --all -- --check && cargo clippy --all-targets --all-features -- -D warnings`
- [ ] `cargo xtask regen-openapi && cargo xtask regen-schema`
- [ ] `cargo run -p tools-spec-extract --quiet && git diff --exit-code`
- [ ] `cargo test --workspace --all-features -- --nocapture`
- [ ] `bash ci/scripts/check_links.sh`

---

## Acceptance Gates (Definition of Done per stage)

- [x] Stage 0 — Contract freeze: OpenAPI + config schema regenerated; CI fails on diffs
- [x] Stage 1 — CDC + snapshots: Pact + insta green before provider code
- [x] Stage 2 — Provider verify: orchestrator passes pact verification
- [x] Stage 3 — Properties: core invariants via proptest
- [x] Stage 4 — Determinism: two replicas per engine; byte‑exact streams
- [x] Stage 5 — Observability: metrics exactly per `.specs/metrics/otel-prom.md`
- [ ] Stage 6 — Admission → Dispatch vertical
- [ ] Stage 7 — Pool manager readiness
- [ ] Stage 8 — Worker adapters conformance
- [ ] Stage 9 — Scheduling & fairness
- [ ] Stage 10 — Capability discovery
- [ ] Stage 11 — Config & quotas
- [ ] Stage 12 — BDD coverage
- [ ] Stage 13 — Dashboards & alerts
- [ ] Stage 14 — Startup self‑tests
- [ ] Stage 15 — Real‑model E2E (Haiku)
- [ ] Stage 16 — Chaos & Load (nightly)
- [ ] Stage 17 — Compliance & Release

---

## Cross‑Cutting Tasks and Hygiene

- [ ] CI: ensure `ci/pipelines.yml` runs fmt, clippy, tests, regen checks, link checks, pact verify
- [ ] CODEOWNERS up‑to‑date for all crates and contracts
- [ ] Security: review `SECURITY.md`, dependency updates, and license headers
- [ ] Idempotent regen tools: second run is diff‑clean (`xtask`, spec‑extract)
- [ ] Ensure all tests and code reference requirement IDs in names and comments
- [ ] Maintain `requirements/*.yaml` linking req → tests → code with coverage notes

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
