# orchestratord — Production Readiness Checklist

This checklist enumerates what `orchestratord` must provide to be production‑ready as the HTTP control and data‑plane daemon of llama‑orch.

Align with the workspace Golden Rules: Spec → Contract → Tests → Code. No back‑compat pre‑1.0. Every user‑visible change ships with spec/contract diffs and tests.

## Scope & References
- Specs: `orchestratord/.specs/00_orchestratord.md`, `orchestratord/.specs/10_orchestratord_v2_architecture.md`
- OpenAPI: `contracts/openapi/{control.yaml,data.yaml}`
- Metrics contract: `ci/metrics.lint.json`
- Related crates: `orchestrator-core`, `pool-managerd`, `worker-adapters/*`

## HTTP Surface & Behavior
- [ ] All documented endpoints implemented and mounted in `src/app/router.rs`
  - [ ] Data: `POST /v1/tasks`, `GET /v1/tasks/{id}/stream`, `POST /v1/tasks/{id}/cancel`
  - [ ] Sessions: `GET /v1/sessions/{id}`, `DELETE /v1/sessions/{id}`
  - [ ] Artifacts: `POST /v1/artifacts`, `GET /v1/artifacts/{id}`
  - [ ] Capabilities: `GET /v1/capabilities` (single source of truth)
  - [ ] Control: `GET /v1/pools/{id}/health`, `POST /v1/pools/{id}/drain`, `POST /v1/pools/{id}/reload`
  - [ ] Observability: `GET /metrics` Prometheus text
- [ ] No legacy endpoints (e.g., `/v1/replicasets`) are served
- [ ] Middleware
  - [ ] `X-API-Key` enforcement as configured
  - [ ] `X-Correlation-Id` echoed or generated on every response, including errors
  - [ ] Request limits (rate/body size) configured per environment

## Admission, Budgets, and Streaming
- [ ] Admission integrates `orchestrator-core` policies; computes real `queue_position`, ETA, and backoff headers
- [ ] Budgets (tokens/time/cost) enforced at admission and reported in headers and SSE metrics frames
- [ ] Streaming path uses adapters with cancel propagation; deterministic event ordering ensured (`started` → `token*` → `metrics?` → `end`)
- [ ] SSE transcript persisted as an artifact; includes engine/pool metadata

## Capabilities & Control Plane
- [ ] `/v1/capabilities` reflects live `AdapterRegistry`/`PoolRegistry` (engines, versions, workloads, ctx_max, concurrency)
- [ ] Control operations are atomic and deadline‑aware; health includes `draining`, queue/replica metrics, and `last_error`

## Artifacts & Storage
- [ ] Artifact store uses CAS with SHA‑256 digests; atomic writes and durable storage
- [ ] Filesystem backend implemented with quotas/GC and robust errors
- [ ] API enforces content size/type; supports conditional requests where useful (ETag)

## Observability & Metrics
- [ ] `/metrics` exposes all required counters/gauges/histograms lint‑clean per `ci/metrics.lint.json`
- [ ] Logs are structured JSON with correlation IDs, including core fields: `job_id`, `session_id`, `engine`, `engine_version`, `pool_id`, `replica_id`, `queue_position`, `predicted_start_ms`, `tokens_in`, `tokens_out`, `decode_time_ms`
- [ ] SSE `metrics` frames include: `queue_depth`, `on_time_probability`, `kv_warmth`, budgets

## Security & Policy
- [ ] API‑key or stronger auth enabled per deployment; secrets never logged
- [ ] Optional policy hooks for HTTP tooling and redaction in logs

## Determinism & Reliability
- [ ] Deterministic mode supported via adapter `seed`; single‑slot path tested with mock
- [ ] Graceful shutdown with in‑flight stream handling and drain semantics
- [ ] Idempotence for task creation where applicable; cancel is race‑free (no tokens after cancel)

## Testing & Quality Gates
- [ ] Unit tests for each API and Service module
- [ ] Provider verification tests green against OpenAPI
- [ ] Local BDD features green for data/control/SSE/security
- [ ] Determinism suite (mock engines) passes; real engine gaps documented
- [ ] Metrics lint passes locally and in CI
- [ ] `cargo fmt` and `clippy -D warnings` clean

## Operations & Packaging
- [ ] Config via env or file for listen addr, auth policy, artifact root, exporters
- [ ] Health/readiness endpoints and logs suitable for service management
- [ ] Release artifacts and versioning documented; CHANGELOG pointers

## Documentation
- [ ] `orchestratord/README.md` High/Mid/Low sections match the code and are kept current
- [ ] `.specs/*` include a “Refinement Opportunities” section with actionable follow‑ups

## Verification Commands
```
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test -p orchestratord -- --nocapture
cargo test -p orchestrator-core -- --nocapture
# Provider verify
target=$(pwd); cargo test -p orchestratord --test provider_verify -- --nocapture
# Metrics lint (workspace script)
bash ci/scripts/check_links.sh || true
```
