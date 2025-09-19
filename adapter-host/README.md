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

Refinement Opportunities
- Add circuit breaker and capability cache.
- Metrics around retries and breaker trips.
- Integration shims into `orchestratord` endpoints.
