# Home Profile Specification (Personal / Single-Workstation)

Purpose: A reduced, practical profile for home labs and single workstations with one or more mixed NVIDIA GPUs. This profile removes enterprise/cluster features while preserving a simple, robust local serving stack.

This is a reductive rewrite of the production SPEC. No new features are added; only removals and simplifications are made.

---

## 1. Scope and Constraints

- MUST target a single host (no cluster orchestration).
- MUST run with mixed GPU generations and VRAM capacities.
- MUST prefer simple, local defaults and minimal configuration.
- MUST prioritize reliability over throughput optimization features.

---

## 2. Security and Access

- MUST bind all HTTP endpoints to 127.0.0.1 by default.
- MUST require a single API token (static) for control and data endpoints.
- SHOULD allow configuring the token via environment variable or config file.
- MUST NOT implement tenant policies, RBAC, or multi‑tenancy isolation.

---

## 3. Admission and Scheduling

- MUST implement a bounded admission queue with two priorities only: `interactive`, `batch`.
- MUST support FIFO within a priority class and one full policy (Reject or Drop‑LRU; implementation‑chosen).
- MUST NOT implement fairness/WFQ, tenant quotas, deadlines (EDF), preemption, or resumable jobs.

---

## 4. Model Lifecycle

- MUST reduce lifecycle states to `Active` and `Retired`.
- MUST NOT use `Draft`, `Deprecated`, `Canary`, or percent rollouts.
- SHOULD support reload of a model atomically; MUST support rollback if reload fails.
- MUST NOT support pool drains or canaries.

---

## 5. Trust and Artifacts

- MAY allow unsigned models and artifacts by default (no mandatory signature/SBOM enforcement).
- SHOULD log warnings when validation artifacts are missing rather than reject.
- MUST NOT expose an artifact registry API in the control plane.

---

## 6. APIs (HTTP)

- MUST provide OrchQueue v1 data plane:
  - POST `/v1/tasks` to admit a task (202 on accept; 400/429/500/503 on error).
  - GET `/v1/tasks/{id}/stream` to stream SSE events: `started`, `token` (repeating), `end`.
- MUST provide a minimal control plane:
  - POST `/v1/pools/{id}/reload` to hot‑reload a model for the pool.
  - GET `/health` for a simple health check.
- MUST NOT expose drain endpoints, artifact registry endpoints, or CDC harnesses.
- MUST NOT emit a correlation ID header.
- MUST NOT include `policy_label` in 429 bodies.

---

## 7. Metrics and Observability

- MUST expose a Prometheus `/metrics` with these required series:
  - `queue_depth{engine,engine_version,pool_id,priority}`
  - `tasks_enqueued_total{engine,engine_version,pool_id,replica_id,priority}`
  - `tasks_rejected_total{engine,reason}`
  - `tokens_in_total{engine,engine_version,pool_id,replica_id}`
  - `tokens_out_total{engine,engine_version,pool_id,replica_id}`
  - `gpu_utilization{engine,engine_version,pool_id,replica_id,device}`
  - `vram_used_bytes{engine,engine_version,pool_id,replica_id,device}`
- SHOULD include a simple model lifecycle gauge if used (`model_state{model_id,state}` with only `Active|Retired`).
- MUST NOT include advanced/enterprise metrics:
  - percentile latencies (p95/p99 labels), deadline ratios, tenant admission share, preempt/resume counters.

---

## 8. Placement

- MUST place work on the least‑loaded GPU based on a simple heuristic (e.g., free VRAM or slot count).
- MUST NOT require NUMA/PCIe topology hints or tensor splitting.

---

## 9. Error Taxonomy (Data Plane)

- MUST keep the existing error envelope shape `{code, message, engine}`.
- MUST use 429 with `Retry-After` and `X-Backoff-Ms` headers on backpressure.
- MUST NOT include `policy_label` advisory field in 429 bodies.

---

## 10. Dropped or Simplified Items (from Production SPEC)

- Multi‑tenancy, RBAC, quotas — removed.
- Fairness (WFQ), deadlines (EDF), preemption, resumable jobs — removed.
- Lifecycle: only `Active|Retired`; drop `Draft|Deprecated|Canary`.
- Trust/SBOM/signatures — optional; enforcement removed.
- Advanced metrics — drop deadline ratios, tenant breakdowns, preempt/resume counters, p95/p99.
- APIs — keep OrchQueue v1 and `/health`. Drop drain, artifact registry, CDC. Remove correlation IDs and 429 policy labels.
- Ops — keep `reload` (+rollback). Drop drains, canaries, percent rollouts, cluster HA.
- Placement — keep least‑loaded only. Drop NUMA/PCIe hints, tensor splitting.
- Security — local only (127.0.0.1), single API token.

---

## 11. Notes for Implementers (Spec→Contracts→Tests→Code)

- Contracts:
  - Remove `/v1/artifacts*` and `/v1/pools/{id}/drain` paths from `contracts/openapi/control.yaml`.
  - Remove correlation ID headers and `policy_label` fields from OpenAPI examples for data plane.
  - Keep only required metrics in `ci/metrics.lint.json` (remove `admission_share`, `deadlines_met_ratio`, `preemptions_total`, `resumptions_total`).
  - Simplify `contracts/config-schema` to drop fairness and preemption structures; keep two priorities only.
- Code:
  - Remove `X-Correlation-Id` usage; remove 429 `policy_label` from `orchestratord/src/http/data.rs`.
  - Remove drains in `orchestratord/src/http/control.rs`; keep `reload` and `/health`.
  - Reduce lifecycle to `Active|Retired` in `orchestratord/src/state.rs` and metrics label values.
  - Delete advanced metrics in `orchestratord/src/metrics.rs` and their uses.
- Tests:
  - Update provider tests to match reduced OpenAPI (no drain, no artifacts, no correlation IDs, no policy_label in 429).
  - Update metrics tests and lint expectations to the reduced set.

This Home Profile supersedes production‑only requirements for local deployments.
