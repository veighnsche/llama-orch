# adapter-host — Component Spec (in-process)

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Purpose & Scope

Provide an in-process registry and facade for `WorkerAdapter` implementations. Centralizes submit/cancel routing, retries/backoff, and capability snapshots without adding network hops.

## Contracts

- Registry API: bind/rebind `(pool_id, replica_id) -> Adapter` on preload/reload.
- Facade: `submit(pool_id, TaskRequest)`, `cancel(pool_id, task_id)`, `health(pool_id)`, `props(pool_id)`.
- Error taxonomy mapping to `WorkerError`.

## Observability

- Narration logs around submit/cancel and error paths.
- Metrics for retries, breaker trips (future).

## Security

- No secrets in logs; redaction via shared util.

## Refinement Opportunities

- Circuit breaker policy and capability cache integration.
- Per-pool timeouts and micro-batch hints surfaced to orchestrator.
