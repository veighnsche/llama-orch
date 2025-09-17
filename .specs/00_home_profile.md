# Llama-Orch SPEC — Home Profile v2.1

Profile: HOME (Personal / Single-Workstation)

Home Profile v2.1 keeps the single-host scope, restores the features the CLI marks as MUST/SHOULD in `cli/llama-orch-cli/feature-requirements.md`, and anchors validation to the reference environment in `.docs/HOME_PROFILE_TARGET.md` (CLI on a dev box, `llama-orch` on a mixed-GPU workstation).

---

## 1. Security & Identity

- MUST bind services to `127.0.0.1` unless overridden.
- MUST accept and echo `X-Correlation-Id` on all HTTP responses.
- MUST authenticate control and data planes with a shared API token; MAY add stronger auth (mTLS/OIDC) but not required.
- MUST provide outbound tool policy hooks (allow/deny lists) even if defaults are permissive.
- MUST NOT implement tenant isolation, RBAC, or quota enforcement.
- SHOULD document how to expose the API to a remote dev box (e.g., SSH tunnel or explicit bind) without weakening defaults.

## 2. Admission & Scheduling

- MUST keep a bounded queue with two priorities: `interactive`, `batch`.
- MUST preserve FIFO within a priority.
- MUST implement a documented full policy (Reject or Drop-LRU).
- MUST accept all FR-DP-001 fields on `POST /v1/tasks`.
- SHOULD include `queue_position` and `predicted_start_ms` in admission responses (nullable when unknown).
- MUST honour determinism knobs (`seed`, `determinism`, `sampler_profile_version`).

## 3. Task Streaming

- `GET /v1/tasks/{id}/stream` MUST emit `started`, repeated `token`, periodic `metrics`, `end`, `error` SSE frames.
- MUST carry `X-Correlation-Id` and budget headers if budgets are enforced.
- MAY coalesce `metrics` frames for efficiency but MUST surface queue depth and token counters.

## 4. Sessions & Budgets

- SHOULD expose `GET/DELETE /v1/sessions/{id}` for KV reuse and eviction.
- SHOULD support per-session token/time budgets with at least advisory enforcement/logging.
- MUST NOT rely on multi-tenant session isolation.

## 5. Control Plane

- MUST keep model catalog endpoints: `POST /v1/catalog/models`, `POST /v1/catalog/models/{id}/verify`, `POST /v1/catalog/models/{id}/state`.
- MUST keep pool endpoints: `POST /v1/pools/{id}/drain`, `POST /v1/pools/{id}/reload`, `GET /v1/pools/{id}/health`.
- MUST support model states `Active` and `Retired`; MAY optionally expose additional states but default workflow uses two.

## 6. Discovery & Versioning

- MUST expose capability discovery via enriched `GET /v1/replicasets` or `GET /v1/capabilities` with engine versions, capacity, max context, rate limits.
- MUST publish API versions in OpenAPI documents (`info.version`).
- SHOULD surface supported sampler profiles and determinism guarantees.

## 7. Artifacts

- SHOULD expose `/v1/artifacts` create/fetch APIs for CLI plan storage.
- MAY allow unsigned artifacts but MUST log trust gaps.
- MUST keep artifact storage local to the home deployment.

## 8. Observability & Metrics

- MUST expose Prometheus with at least:
  - `queue_depth{engine,engine_version,pool_id,priority}`
  - `tasks_enqueued_total{engine,engine_version,pool_id,replica_id,priority}`
  - `tasks_rejected_total{engine,reason}`
  - `tokens_in_total{engine,engine_version,pool_id,replica_id}`
  - `tokens_out_total{engine,engine_version,pool_id,replica_id}`
  - `gpu_utilization{engine,engine_version,pool_id,replica_id,device}`
  - `vram_used_bytes{engine,engine_version,pool_id,replica_id,device}`
- SHOULD expose `model_state{model_id,state}` with states limited to `Active|Retired`.
- MAY omit percentile histograms and tenant/fairness gauges.
- MUST log warnings instead of failing when SBOM/signature data is missing.

## 9. Errors & Backpressure

- MUST keep error envelope `{code, message, engine, correlation_id?}`.
- MUST send HTTP 429 with `Retry-After`, `X-Backoff-Ms`, and `policy_label` in the body.
- SHOULD document retry guidance consistent with headers.

## 10. Placement & Performance

- MUST schedule on least-loaded GPU (free VRAM/slots heuristic) without requiring NUMA/tensor hints.
- MUST publish capacity so the CLI can derive safe concurrency (FR-MA-001).
- SHOULD provide `predicted_start_ms` estimates (FR-PF-002).
- MUST validate placement against mixed RTX 3090 (24 GB) and RTX 3060 (12 GB) cards.

## 11. Differences vs Production

- Removed: multi-tenant quotas, fairness (WFQ), preemption, resumable jobs, distributed control plane.
- Retained (relative to v1 reduction): catalog, drain, artifacts, capability discovery, correlation IDs, SSE metrics frames, `policy_label`, admission metadata.

## 12. Spec→Contracts→Tests→Code Alignment

- Contracts must include catalog, drain, artifacts, capability discovery, SSE `metrics`, correlation IDs, `policy_label`, determinism fields, and admission metadata.
- Config schema keeps two priorities but MUST retain determinism/session/budget fields consumed by the CLI.
- Metrics lint MUST match the metric list above; optional series remain optional.
- Provider tests and Pacts MUST cover the reinstated endpoints and headers.
- Code must echo `X-Correlation-Id`, honour determinism, implement catalog/drain/artifact flows, and bind to loopback by default.
- Validation suites SHOULD exercise the reference deployment described in `.docs/HOME_PROFILE_TARGET.md` before release.

Conformance to Home Profile v2.1 supersedes the initial HOME profile reduction. Do not remove surfaces marked as MUST/SHOULD for CLI compatibility.
