# Worker Adapters — Contracts

Status: Draft
Owner: @llama-orch-maintainers
Date: 2025-09-19

## Provided Contracts (from adapters to orchestrator)

- `WorkerAdapter` trait implementation per adapter with:
  - Health/readiness: `health() -> WorkerHealth { live, ready }`.
  - Capacity/props: `props() -> WorkerProps { slots_total?, slots_free? }`.
  - Streaming: `submit(TaskRequest) -> TokenStream` emitting `started` → `token*` → `end` (and optional `metrics`) or terminal `error`.
  - Cancellation: `cancel(task_id)` best-effort.
  - Versioning: `engine_version() -> String`.
- Error mapping to `WorkerError` taxonomy.
- Required log fields (README_LLM) and metrics (otel-prom.md).

## Consumed Contracts (from orchestrator and others)

- Tasks: `contracts/api-types::TaskRequest`.
- Control: `orchestratord` may assert timeouts/cancellations and backpressure decisions; adapters should respect them.
- Health topology: `pool-managerd` will include adapter health/props in pool readiness summaries.
- Model/artifacts: engine command-lines or HTTP endpoints will point at `ResolvedModel.local_path` created by provisioners; adapters should not stage models.

## Data Exchange

- Token event frames and their minimal schemas (`started`, `token`, `metrics`, `end`, `error`).
- Health/props snapshots used by `orchestratord` and placement.

## Versioning & Compatibility

- Trait is stable within workspace; adding new fields/kinds must be feature-gated or coordinated.
- Adapter behavior should be consistent across engines; deviations are documented in per-adapter specs.

## Observability

- Metrics: counters (requests, tokens), histograms (latencies), gauges (slots); label budgets enforced.
- Logs: include standard fields (job_id, session_id, engine, engine_version, pool_id, replica_id, queue_position, predicted_start_ms, tokens_in, tokens_out, decode_time_ms).

## Security & Policy

- Redact secrets (API keys, tokens); respect network egress and timeouts.
- Do not leak full prompts unless policy allows; sampling for dev only.

## Testing Expectations

- Per-adapter unit/behavior tests for mapping and retries.
- Root BDD covers cross-crate flows only.

## Refinement Opportunities

- Shared HTTP/retry/stream decode utility crate to reduce duplication across adapters.
- Capability schema and discovery handshake with `pool-managerd`.
