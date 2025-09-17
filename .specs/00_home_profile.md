# Llama-Orch SPEC — Home Profile

Profile: HOME (Personal / Single-Workstation)

This profile reduces the production, multi-tenant specification to a simple local-serving model for single hosts with one or more mixed NVIDIA GPUs. No new features are introduced. Only removals and simplifications are allowed.

Conformance keywords: MUST, SHOULD, MAY.

---

## 1. Security and Access

- MUST bind all HTTP endpoints to 127.0.0.1 by default.
- MUST require a single API token for control and data planes.
- SHOULD allow configuring the token via environment or config file.
- MUST NOT implement tenant policies, RBAC, or multi‑tenancy.

## 2. Admission and Scheduling

- MUST implement a bounded queue with two priorities only: `interactive`, `batch`.
- MUST preserve FIFO within the same priority.
- MUST implement one full policy (Reject or Drop‑LRU; implementation‑chosen) and document it.
- MUST NOT include WFQ/fairness, quotas, deadlines (EDF), preemption, or resumable jobs.

## 3. Lifecycle

- MUST limit lifecycle to `Active` and `Retired` states.
- MUST NOT include `Draft`, `Deprecated`, `Canary`, or percent rollouts.
- SHOULD support atomic reload with rollback on failure.
- MUST NOT expose pool drains or canaries.

## 4. Trust/Artifacts

- MAY allow unsigned artifacts by default; MUST NOT enforce signatures/SBOM.
- SHOULD warn on missing validation artifacts rather than reject.
- MUST NOT expose artifact registry APIs.

## 5. HTTP APIs

- MUST provide OrchQueue v1:
  - POST `/v1/tasks` → 202 accept; 400/429/500/503 error mapping
  - GET `/v1/tasks/{id}/stream` → SSE events: `started`, `token` (repeating), `end`
- MUST provide minimal control plane:
  - POST `/v1/pools/{id}/reload`
  - GET `/health`
- MUST NOT expose drain endpoints, artifact registry, or CDC harnesses.
- MUST NOT emit correlation IDs; responses SHOULD be minimal.
- MUST NOT include `policy_label` in 429 bodies.

## 6. Metrics and Observability

- MUST expose Prometheus metrics limited to:
  - `queue_depth{engine,engine_version,pool_id,priority}`
  - `tasks_enqueued_total{engine,engine_version,pool_id,replica_id,priority}`
  - `tasks_rejected_total{engine,reason}`
  - `tokens_in_total{engine,engine_version,pool_id,replica_id}`
  - `tokens_out_total{engine,engine_version,pool_id,replica_id}`
  - `gpu_utilization{engine,engine_version,pool_id,replica_id,device}`
  - `vram_used_bytes{engine,engine_version,pool_id,replica_id,device}`
- SHOULD expose `model_state{model_id,state}` with values limited to `Active|Retired`.
- MUST NOT expose: fairness/tenant gauges, deadline ratios, preempt/resume counters, p95/p99 histograms.

## 7. Placement

- MUST place on the least‑loaded GPU (e.g., free VRAM or slots) with a simple heuristic.
- MUST NOT depend on NUMA/PCIe topology hints or tensor splitting.

## 8. Error Taxonomy (Data Plane)

- MUST keep `{code, message, engine}` error envelope.
- MUST use 429 with `Retry-After` and `X-Backoff-Ms` headers on backpressure.
- MUST NOT include `policy_label` in 429 bodies.

## 9. Differences vs Production Profile

- Multi‑tenancy/RBAC/quotas — removed.
- Scheduling: fairness, deadlines, preemption, resumable jobs — removed.
- Lifecycle: only `Active|Retired` (no `Draft|Deprecated|Canary`).
- Trust/SBOM/signatures — optional; enforcement removed.
- Metrics: only queue depth, rejects, tokens in/out, GPU/VRAM; advanced metrics removed.
- APIs: keep OrchQueue v1 and `/health`; remove drains, artifacts, CDC; no correlation IDs or 429 policy labels.
- Ops: keep `reload` (+rollback); remove drains, canaries, HA.
- Placement: least‑loaded heuristic only; remove topology/tensor hints.

## 10. Spec→Contracts→Tests→Code Alignment

- OpenAPI: remove `/v1/artifacts*` and drain endpoints; drop correlation ID headers and 429 `policy_label` fields.
- Config schema: remove fairness, preemption, tenant quotas; keep two priorities only.
- Metrics lint (`ci/metrics.lint.json`): keep only required series listed above; remove `admission_share`, `deadlines_met_ratio`, `preemptions_total`, `resumptions_total`, latency histograms.
- Code: remove correlation ID emission; simplify lifecycle to `Active|Retired`; remove drains/artifacts; keep reload.
- Tests: update provider tests and metrics tests to match the reduced profile.

Conformance to this HOME profile supersedes production-only requirements for local deployments.
