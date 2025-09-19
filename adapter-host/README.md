# adapter-host — In-Process Adapter Registry & Facade

Status: Draft
Owner: @llama-orch-maintainers

Purpose
- Registry keyed by `(pool_id, replica_id) -> Box<dyn WorkerAdapter>`.
- Facade: `submit(pool, TaskRequest)`, `cancel(pool, task_id)`, `health`, `props`.
- Centralize retries/backoff, timeouts, basic circuit breaker, cancellation routing.
- Capability snapshot cache for `/v1/capabilities`.
- Human narration wrappers (actor=adapter-host, action=submit|cancel, target=engine).

Links
- Trait: `worker-adapters/adapter-api`
- Spec: `.specs/proposals/2025-09-19-adapter-host-and-http-util.md` (ORCH-36xx)

Detailed behavior (High / Mid / Low)

- High-level
  - Exposes an in-process facade over engine adapters for `orchestratord`, hiding per-adapter client setup and providing a stable API: `health`, `props`, `submit`, `cancel`, `engine_version`.
  - Maintains a thread-safe registry keyed by `(pool_id, replica_id)` and supports rebinding during drain/reload.

- Mid-level
  - Wraps adapter calls with narration and metrics; emits fields per `README_LLM.md` and metrics per `.specs/metrics/otel-prom.md`.
  - Propagates `Authorization: Bearer` to orchestrator endpoints when configured; redacts secrets in logs.
  - Applies bounded retries with jitter for idempotent operations; enforces per-request timeouts; routes cancellation promptly.
  - Caches capability snapshots per adapter to serve `/v1/capabilities` quickly while respecting TTLs.

- Low-level
  - Registry updates are lock-scoped and minimize contention; adapters are stored behind `Arc<dyn WorkerAdapter + Send + Sync>`.
  - Correlation IDs are threaded through all calls; errors are normalized to the shared `WorkerError` taxonomy for consistent envelopes.
  - Cancel paths prefer in-flight cancellation signaling over connection teardown, and guarantee no tokens after cancel.

Refinement Opportunities
- Add circuit breaker and capability cache.
- Metrics around retries and breaker trips.
- Integration shims into `orchestratord` endpoints.
