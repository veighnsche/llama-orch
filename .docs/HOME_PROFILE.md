# Home Profile Specification (Single-Host Home Lab)

This document defines the only supported deployment profile for llama-orch: a single workstation with NVIDIA GPUs, driven by a developer box over the local network. There is no “enterprise edition” to fall back to, and no reduction narrative—this is the baseline product.

For topology details see `.docs/HOME_PROFILE_TARGET.md`. Contracts are captured formally in `.specs/00_home_profile.md` and the component specs under `.specs/**`.

---

## 1. Goals and Constraints

- MUST run all orchestrator services on a single host with one or more NVIDIA GPUs (mixed VRAM is expected).
- MUST allow the developer box to drive the system remotely (SSH tunnel or explicit bind) while defaulting to loopback-only exposure.
- MUST keep configuration lightweight: filesystem-backed storage, single API token, no external control plane.
- SHOULD prioritise determinism, observability, and quick debugging over throughput tuning.
- MUST keep the Spec → Contract → Tests → Code workflow intact (see `.docs/PROCESS.md`).

---

## 2. Interfaces (Summary)

### Data Plane (OrchQueue v1)

- `POST /v1/tasks` — admits a task with fields for `task_id`, `session_id`, `workload`, `model_ref`, `engine`, `ctx`, `priority`, determinism knobs (`seed`, `determinism`, `sampler_profile_version`), prompts/inputs, `max_tokens`, `deadline_ms`, optional session hints.
- `GET /v1/tasks/{id}/stream` — Server-Sent Events in the fixed order `started`, `token` (repeating), `metrics` (optional, additive), `end`, `error`.
- `POST /v1/tasks/{id}/cancel` — idempotent cancel for queued or running work.
- `GET/DELETE /v1/sessions/{id}` — lightweight session inspection and eviction for KV reuse.

### Control Plane

- Catalog: `POST /v1/catalog/models`, `GET /v1/catalog/models/{id}`, `POST /v1/catalog/models/{id}/verify`, `POST /v1/catalog/models/{id}/state` (supports `Active`/`Retired`).
- Pools: `POST /v1/pools/{id}/drain`, `POST /v1/pools/{id}/reload`, `GET /v1/pools/{id}/health`.
- Discovery: either enriched `GET /v1/replicasets` or dedicated `GET /v1/capabilities` to surface engine limits, concurrency hints, and versioning.

### Observability

- `/metrics` — Prometheus exposition with the minimal series listed in §5.
- Structured JSON logs with correlation IDs.

All APIs require a single shared API token. Endpoints bind to `127.0.0.1` by default and can be exposed over LAN deliberately.

---

## 3. Functional Requirements

### Admission & Queueing
- MUST keep exactly two priorities (`interactive`, `batch`) and execute FIFO within each priority.
- MUST implement one bounded-queue policy (`reject` or `drop-lru`) and document which policy is active.
- MUST return `queue_position`, `predicted_start_ms`, and `backoff_ms` in the admission response when that information is available; MAY return `null`/`0` placeholders if estimation is not yet implemented.
- MUST emit proper backpressure signals on 429: `Retry-After`, `X-Backoff-Ms`, and a JSON body containing `policy_label`, `retriable`, `retry_after_ms`.

### Streaming & Determinism
- SSE stream MUST emit correlation IDs and optional budget headers once per stream.
- Determinism controls (`seed`, `determinism`, `sampler_profile_version`) MUST pass through to worker adapters and produce repeatable streams on the same replica.
- `metrics` frames MAY appear multiple times and carry additive JSON data (queue depth, estimated on-time probability, budgets, KV warmth). Omit the frame entirely only if there is no useful data.

### Sessions & Budgets
- Sessions are short-lived (default TTL ≤ 10 minutes) with at most 8 turns; session metadata MUST expose TTL, turns, KV usage, and optional budget remaining.
- Token/time/cost budgets MAY be advisory; if enforcement is implemented it MUST reject before enqueue and surface the remaining budget in headers or SSE metrics.

### Catalog & Artifact Handling
- Catalog entries MUST persist model metadata locally (filesystem or sqlite) and allow verification that may downgrade unsigned uploads to warnings rather than failures.
- Artifact registry (optional but recommended) SHOULD provide `POST /v1/artifacts` + `GET /v1/artifacts/{id}` to store agent plans, diffs, and traces with content-addressable IDs. Storage is local to the workstation.

### Placement & GPU Scheduling
- Placement MUST choose the least-loaded GPU using a simple heuristic (free VRAM or available slots) and respect explicit device masks.
- Mixed VRAM scheduling (e.g., RTX 3090 + 3060) MUST work without manual per-model pinning; validation happens against the reference environment.

---

## 4. Security & Policy
- Control and data plane MUST share an API token (loaded via environment or config file). Optional mTLS/OIDC can be added later but is not required for baseline.
- All HTTP responses MUST echo or generate an `X-Correlation-Id` header.
- Placement and tool invocations MUST flow through a simple policy hook that can allow/deny outbound HTTP actions (defaults permissive, configurable per home lab).
- Logs MUST redact secrets and tokens.

---

## 5. Observability Contract (Minimum)

- `queue_depth{engine,engine_version,pool_id,priority}` — gauge.
- `tasks_enqueued_total{engine,engine_version,pool_id,replica_id,priority}` — counter.
- `tasks_rejected_total{engine,reason}` — counter (reason includes `ADMISSION_REJECT`, `QUEUE_FULL_DROP_LRU`, `INVALID_PARAMS`, `POOL_UNAVAILABLE`).
- `tasks_started_total{engine,engine_version,pool_id,replica_id,priority}` and `tasks_canceled_total{engine,engine_version,pool_id,replica_id,reason}` — counters.
- `tokens_in_total{engine,engine_version,pool_id,replica_id}` and `tokens_out_total{engine,engine_version,pool_id,replica_id}` — counters.
- `gpu_utilization{engine,engine_version,pool_id,replica_id,device}` and `vram_used_bytes{engine,engine_version,pool_id,replica_id,device}` — gauges.
- Optional gauges: `model_state{model_id,state}` (`Active|Retired` only), `kv_cache_usage_ratio`.

Log events for admission MUST include `queue_position`, `predicted_start_ms`, `engine`, `engine_version`, `pool_id`, and `replica_id`.

---

## 6. Testing Expectations
- Spec changes MUST follow `.docs/PROCESS.md` (Spec → Contract → Tests → Code).
- Provider verification (`orchestratord/tests/provider_verify.rs`) MUST stay green for every API change.
- BDD suite (`test-harness/bdd`) MUST cover admitted tasks, streaming order, cancel, catalog interactions, artifact persistence, and mixed-GPU scheduling.
- Determinism suite MUST run on two replicas per engine with fixed seeds.
- Reference environment smoke test MUST pass before release; see `.docs/HOME_PROFILE_TARGET.md`.

---

## 7. Implementation Notes
- Contracts live in `contracts/openapi/{data.yaml,control.yaml}` and `contracts/config-schema/src/lib.rs`.
- Metrics lint (`ci/metrics.lint.json`) must match the series above.
- Configuration examples live in `examples/home-profile/` (to be created alongside implementation work).
- TODO tracker updates are mandatory after every doc/spec/code/test change.

Use this document as the narrative overview. Normative statements and requirement IDs reside in the `.specs/` tree.
