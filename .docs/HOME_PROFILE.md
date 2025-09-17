# Home Profile Specification v2.1 (Personal / Single-Workstation)

Purpose: Deliver a practical, single-host profile that stays compatible with the CLI requirements in `cli/llama-orch-cli/feature-requirements.md` while still trimming cluster-only features. Version 2.1 aligns the spec with the reference environment described in `.docs/HOME_PROFILE_TARGET.md` (dev box running the CLI, workstation hosting `llama-orch` with RTX 3090 + 3060 GPUs).

Home Profile v2.1 remains a reductive rewrite of the production SPEC. Any additions below are reinstated because the CLI marks them as MUST (or high-value SHOULD) requirements. Where we still diverge from production we call it out explicitly.

---

## 1. Scope and Constraints

- MUST target a single host (no multi-node orchestration).
- MUST support mixed NVIDIA GPUs and VRAM capacities.
- MUST default to local-only operation with minimal external dependencies.
- SHOULD prioritize deterministic behaviour and debuggability over throughput tuning.
- MUST stay interoperable with the CLI’s Spec→Contract→Tests→Code loop.
- SHOULD validate end-to-end behaviour against the reference workstation/CLI pairing documented in `.docs/HOME_PROFILE_TARGET.md`.

---

## 2. Identity, Security, and Policy

- MUST bind HTTP services to `127.0.0.1` by default; MAY expose additional interfaces via explicit configuration (e.g., for the dev-box→workstation tunnel).
- MUST support a single API token for developer usage; MAY optionally support mTLS/OIDC, but NOT required for HOME.
- MUST accept `X-Correlation-Id` and echo it in all responses (FR-OB-001).
- MUST provide policy hooks for outbound tooling (HTTP fetch/search) even if default policies are permissive (FR-TL-002).
- MUST NOT implement tenant isolation or quota enforcement.

---

## 3. Data Plane — Task Admission & Streaming

- MUST keep OrchQueue v1 endpoints:
  - POST `/v1/tasks` accepts fields required by FR-DP-001 (`task_id`, `session_id`, `workload`, `model_ref`, `engine`, `ctx`, `priority`, `seed`, `determinism`, `sampler_profile_version`, `prompt|inputs`, `max_tokens`, `deadline_ms`, optional `kv_hint`).
  - Response MUST include admission metadata when available (`queue_position`, `predicted_start_ms`) per FR-DP-005 (SHOULD) — MAY fall back to `null` if estimates are unavailable.
  - GET `/v1/tasks/{id}/stream` emits SSE frames: `started`, repeated `token`, periodic `metrics`, `end`, `error` (FR-DP-002, FR-OB-003).
- MUST continue to support two priorities (`interactive`, `batch`) and a bounded queue (retains earlier simplification).
- MUST keep FIFO ordering within a priority; MAY choose Reject or Drop-LRU policy but MUST document the behaviour.
- MUST support deterministic execution flags (`seed`, `determinism`) and honour them (FR-DE-001).

---

## 4. Sessions, KV Reuse, and Budgets

- SHOULD expose `GET /v1/sessions/{id}` and `DELETE /v1/sessions/{id}` for KV reuse/cleanup (FR-DP-004).
- SHOULD support per-session token/time budgets with advisory enforcement or logging (FR-MA-002). Hard enforcement MAY be deferred but MUST be documented if absent.
- MAY omit multi-tenant budgeting/grants; keep scope to single user workflows.

---

## 5. Control Plane — Catalog & Pools

- MUST retain the model catalog endpoints (FR-CP-001):
  - `POST /v1/catalog/models`
  - `POST /v1/catalog/models/{id}/verify`
  - `POST /v1/catalog/models/{id}/state`
- MUST keep pool lifecycle endpoints (FR-CP-002):
  - `POST /v1/pools/{id}/drain`
  - `POST /v1/pools/{id}/reload`
  - `GET /v1/pools/{id}/health`
- MUST support control plane auth using the same token; MUST emit `X-Correlation-Id`.
- MUST keep lifecycle states `Active`, `Retired`; MAY expose additional state transitions if required for CLI compatibility but default to two states.

---

## 6. Discovery & Versioning

- MUST expose capability discovery (FR-DV-001) via either:
  - `GET /v1/replicasets` enriched with engine versions, max context, rate limits, estimated concurrency, **or**
  - `GET /v1/capabilities` providing the same data.
- MUST signal API versions in OpenAPI `info.version` (FR-DV-002).
- SHOULD document supported sampler profiles and determinism guarantees alongside capability discovery.

---

## 7. Artifact Registry and Lineage

- SHOULD provide `/v1/artifacts` (create + fetch) so the CLI can persist plans, diffs, and traces (FR-AR-001).
- MUST allow unsigned artifacts by default but SHOULD log verification gaps.
- MUST keep artifacts local (e.g., filesystem or sqlite) — no distributed registry requirement.

---

## 8. Observability & Telemetry

- MUST keep Prometheus `/metrics` endpoint with at least:
  - `queue_depth{engine,engine_version,pool_id,priority}`
  - `tasks_enqueued_total{engine,engine_version,pool_id,replica_id,priority}`
  - `tasks_rejected_total{engine,reason}`
  - `tokens_in_total{engine,engine_version,pool_id,replica_id}`
  - `tokens_out_total{engine,engine_version,pool_id,replica_id}`
  - `gpu_utilization{engine,engine_version,pool_id,replica_id,device}`
  - `vram_used_bytes{engine,engine_version,pool_id,replica_id,device}`
- SHOULD expose `model_state{model_id,state}` with values `Active|Retired`.
- MAY include additional per-request or latency metrics if inexpensive; percentile histograms remain OPTIONAL.
- MUST emit correlation IDs (`X-Correlation-Id`) and MUST NOT drop them in Home profile responses.
- MUST keep SSE `metrics` frames in the task stream (FR-OB-003).
- SHOULD log warnings instead of hard failures for missing SBOM/signature data.

---

## 9. Errors & Backpressure

- MUST keep the canonical error envelope `{code, message, engine, correlation_id?}`.
- MUST return HTTP `429` with `Retry-After` (seconds) and `X-Backoff-Ms` headers plus `policy_label` advisory fields (FR-DP-003, FR-ER-002).
- SHOULD document retry/backoff guidance aligned with these headers.

---

## 10. Placement, Concurrency & Performance

- MUST schedule using a simple least-loaded GPU heuristic (free VRAM/slots) that fits single-host deployments.
- MUST derive safe concurrency from capacity data exposed in capability discovery (FR-MA-001, FR-PF-001).
- SHOULD surface `predicted_start_ms` where possible (FR-PF-002).
- MUST NOT require NUMA-aware placement or tensor parallelism; validation must cover mixed-GPU (RTX 3090 + 3060) scheduling.

---

## 11. Differences vs Production Profile

- Retained reductions:
  - Multi-tenancy, RBAC, complex quota systems remain out of scope.
  - Fairness scheduling (WFQ), preemption, resumable jobs, deadline scheduling remain removed.
  - Distributed artifact registry, HA control plane, CDC pipelines remain out of scope.
- Restored items (relative to v1) because the CLI marks them as MUST/SHOULD:
  - Model catalog and pool drain endpoints.
  - Artifact registry (SHOULD), capability discovery, health endpoints.
  - Correlation IDs, SSE `metrics` frames, `policy_label` in 429 responses.
  - Admission metadata (`queue_position`, `predicted_start_ms`) when available.

---

## 12. Notes for Implementers (Spec→Contracts→Tests→Code)

- Contracts:
  - Ensure `contracts/openapi/*` retains catalog, drain, artifacts, capability discovery, and correlation ID fields.
  - Update examples to include `policy_label`, SSE `metrics`, admission metadata, and determinism parameters.
  - Keep `ci/metrics.lint.json` aligned with the metric list above; optional histograms remain suppressed.
- Code:
  - Guarantee `X-Correlation-Id` echoing and backpressure headers.
  - Honour determinism knobs and restore SSE `metrics` emission.
  - Keep catalog, drain, and artifact handlers minimal yet functional.
  - Continue to default bind addresses to loopback and enforce simple token auth.
- Tests:
  - Provider verification and Pact fixtures must reflect the reinstated surfaces.
  - BDD scenarios should cover catalog interactions, drains, SSE `metrics`, and artifact flows.
  - Determinism, queue metadata, and policy enforcement require explicit tests.

Home Profile v2.1 supersedes the initial reduction. Use this document as the reference when trimming code paths; do not remove items listed as required for CLI compatibility.
