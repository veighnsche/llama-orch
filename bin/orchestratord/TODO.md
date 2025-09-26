# orchestratord — Single TODO (Spec‑first tracker)

Status: active
Last updated: 2025-09-19

Keep this the single source of truth for pending work in the `orchestratord` crate. Follow the workspace guidelines: Spec → Contract → Tests → Code. Any user‑visible behavior change must update specs/contracts and ship with tests.

## 0) Scope & References
- Specs
  - `orchestratord/.specs/00_orchestratord.md`
  - `orchestratord/.specs/10_orchestratord_v2_architecture.md`
  - Workspace: `.specs/00_llama-orch.md`, `.specs/00_home_profile.md`
- Contracts: `contracts/openapi/{control.yaml,data.yaml}`
- Metrics contract: `ci/metrics.lint.json`
- Tests to run: `cargo test -p orchestratord -- --nocapture`

## 1) App Layer
- [ ] `src/app/bootstrap.rs`: graceful shutdown (SIGINT/SIGTERM) and draining semantics.
- [ ] Inject constructed state (AdapterRegistry, ArtifactStore, PoolRegistry, config) in a single bootstrap path.
- [ ] Startup logs with version, enabled features, and metrics endpoint.
- [ ] `src/app/router.rs`: add a unit test that asserts mounted routes match OpenAPI (no extras).
- [ ] `src/app/middleware.rs`: make API‑key policy configurable; add rate limits and body size limits; ensure `X-Correlation-Id` on every response (including errors).

## 2) API Layer
- [ ] `src/api/data.rs`: integrate `admission::QueueWithMetrics` for enqueue; compute real `queue_position`, ETA, and backoff headers.
- [ ] Budget headers: derive from session/budget policy; remove sentinel shortcuts.
- [ ] Streaming: wire to `ports::adapters::AdapterClient` with async token stream and cancel propagation; migrate to proper SSE/chunked streaming.
- [ ] Logging: add structured logs with correlation IDs for enqueue/start/end/error.
- [ ] `src/api/control.rs`: atomic drain/reload via `PoolRegistry` with deadlines/rollback; health to include real queue/replica metrics and `last_error`.
- [ ] `src/api/artifacts.rs`: depend on `ports::storage::ArtifactStore`; enforce content limits; add ETag/If‑None‑Match where useful.
- [ ] `src/api/observability.rs`: gather from real registry; include build info; optionally add `/healthz` and `/readyz`.

## 3) Services
- [ ] `services/session.rs`: replace fixed tick with last_seen bookkeeping; enforce tokens/time/cost budgets; emit session metrics (evictions, counts).
- [ ] `services/streaming.rs`: record tokens_in/out and latency from the real adapter stream; ensure ordering and include admission context in `started`.
- [ ] `services/capabilities.rs`: build snapshot dynamically from `AdapterRegistry`/`PoolRegistry` (ctx_max, workloads, concurrency, versions).
- [ ] `services/artifacts.rs`: façade over `ArtifactStore` with indexing hooks for transcripts and search.
- [ ] `services/control.rs`, `services/catalog.rs`: implement orchestration and trust hooks; keep API thin.

## 4) Ports & Infra
- [ ] `ports/adapters.rs`: expand workloads (completion/embedding/rerank), properties, and async streaming API with backpressure + cancellation.
- [ ] `ports/pool.rs`: add replicas/load/health metrics, timeouts, adapter discovery per pool, model mapping, and drain progress.
- [ ] `ports/storage.rs`: streaming/attachment interfaces and metadata.
- [ ] `infra/storage/fs.rs`: CAS layout, atomic writes, checksum verification, GC/quotas, robust errors.
- [ ] `infra/metrics.rs`: Prometheus/OTel exporter (feature‑gated) replacing the local shim.
- [ ] `infra/clock.rs`: test clock with drift controls.

## 5) Admission & Metrics
- [ ] `admission.rs`: integrate budgeting (tokens/time/cost) and priority queues; emit backpressure metrics with taxonomy labels.
- [ ] `metrics.rs`: replace shim with exporter‑backed registry; enforce label cardinality budgets; unit tests for labels.

## 6) Domain & State
- [ ] `domain/error.rs`: populate `engine` dynamically from selected adapter/pool; finalize correlation‑id policy (headers vs body).
- [ ] `domain/sse.rs`: align event payloads + serializer helpers to adapter outputs.
- [ ] `domain/ids.rs`: parsing/validation from paths/headers.
- [ ] `state.rs`: use `RwLock` where read‑heavy; add background TTL eviction and drain progression.

## 7) Binary/Lib Surface
- [ ] `main.rs`: graceful shutdown, env/config for listen addr/auth/features; startup banner; run with defaults.
- [ ] `lib.rs`: re‑export public API intentionally; crate‑level docs linking `.specs/` and OpenAPI.

## 8) Cross‑Cutting
- [ ] Remove all sentinel shortcuts (e.g., `prompt == "cause-internal"`); replace with real orchestration.
- [ ] Guarantee `X-Correlation-Id` present on all responses (incl. errors) via middleware.
- [ ] Rate limiting and body limits per spec/UX goals.
- [ ] Metrics parity with `ci/metrics.lint.json`; add missing series and labels.
- [ ] Verify OpenAPI coverage for all paths/fields with provider tests.

## 9) Testing & Quality Gates
- [ ] Unit tests per API/Service module.
- [ ] Provider verification (`tests/provider_verify.rs`) kept green after any contract change.
- [ ] Local BDD features for data/control/SSE/security kept green.
- [ ] Determinism suite for mock adapter; document real engine gaps.
- [ ] Metrics lint passes; add tests for label budgets.

## 10) Documentation
- [ ] Keep `orchestratord/README.md` High/Mid/Low sections code‑grounded and current.
- [ ] Propagate High/Mid/Low behavior docs across related crates’ READMEs.
- [ ] Add “Refinement Opportunities” sections to `./.specs/*.md` with actionable follow‑ups.

## 11) Milestones (reference)
- Phase 1: Scaffolding/middleware/tests
- Phase 2: Capabilities/session/health
- Phase 3: Admission/SSE/cancel/backpressure
- Phase 4: Artifacts + transcript capture
- Phase 5: Drain/reload/error mapping/metrics
- Phase 6: Determinism/budgets/hardening

## 12) Verification Commands
```
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --workspace --all-features -- --nocapture
# Provider verify (focused)
cargo test -p orchestratord --test provider_verify -- --nocapture
```
