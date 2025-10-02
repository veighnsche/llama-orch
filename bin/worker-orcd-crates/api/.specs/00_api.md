# Worker API SPEC — RPC Protocol & HTTP Endpoints (WORKER-4xxx)

**Status**: Draft  
**Applies to**: `bin/worker-orcd-crates/api/`  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Scope

This crate implements the HTTP/RPC server for worker-orcd, exposing Plan/Commit/Ready/Execute endpoints with authentication and SSE streaming.

**Parent spec**: `bin/worker-orcd/.specs/00_worker-orcd.md`

---

## 1. Endpoint Authentication

- [WORKER-4200] All RPC endpoints MUST require Bearer token authentication (except `/health` for liveness probes).
- [WORKER-4201] Workers MUST use timing-safe token comparison via `auth-min` crate primitives.
- [WORKER-4202] Workers MUST log identity breadcrumbs (token fingerprint fp6) for all authenticated requests.
- [WORKER-4203] Workers MUST reject requests with invalid or missing tokens with HTTP 401 Unauthorized.

---

## 2. Plan Endpoint

```
POST /worker/plan
```

- [WORKER-4210] The Plan endpoint MUST determine feasibility of loading a model given VRAM constraints.
- [WORKER-4211] Request MUST include: `model_ref`, `shard_layout` (`single` | `tensor_parallel`), `tp_degree` (if TP).
- [WORKER-4212] Response MUST include: `feasible: bool`, `vram_required: usize`, `shard_plan: Vec<ShardPlan>`.
- [WORKER-4213] Plan MUST check Model Capability Descriptor (MCD) against Engine Capability Profile (ECP) and reject if incompatible.
- [WORKER-4214] Plan MUST validate that `vram_required` does not exceed available VRAM on target GPU(s).

### 2.1 Planning Enums (non-breaking)

To standardize planning across components, workers MAY accept an optional `intent` and MAY return a structured `decision`. These are additive and MUST NOT break existing clients.

Request (optional field):
```json
{
  "model_ref": "...",
  "shard_layout": "single",
  "tp_degree": null,
  "intent": { "type": "StageVRAM" }
}
```

Decision (optional fields in response):
```json
{
  "feasible": true,
  "vram_required": 123456,
  "shard_plan": [ ... ],
  "decision": { "type": "CommitVRAM", "reason": "fits", "est_bytes": 123456 }
}
```

Canonical enum variants (serde adjacently-tagged with `type`):

- PlanIntent (request, optional):
  - `ReuseResident { handle_id }`
  - `StageVRAM {}`
  - `EvictThenStage { target_bytes?: u64 }`
  - `PrefetchToRAM { ttl_ms?: u32 }` (pool-local optimization hint)
  - `EstimateOnly {}` (no side effects)
  - `DrainCheck {}` (expect gating status)

- PlanDecision (response, optional):
  - `UseResident { handle_id, slots_free, executing_count?: u32, executing_for_ms?: u32, queue_depth?: u32, est_slot_eta_ms?: u32 }`
  - `CommitVRAM { est_bytes, shard_plan, est_commit_eta_ms: u32, can_execute_now: bool }`
  - `EvictThenCommitVRAM { shard_ids: [String], bytes_to_free: u64, est_eviction_eta_ms: u32, est_commit_eta_ms: u32, can_execute_now_after: bool }`
  - `CannotFitVRAM { required: u64, available: u64, queue_depth?: u32 }`
  - `Draining {}`
  - `EvictSuggested { shard_ids: [String], bytes_to_free: u64 }`
  - `PrefetchRecommended { path?: String, est_bytes: u64, trigger?: { "type": "AfterJobs", "count": u32 } }`
  - `UnsupportedCapabilities { reasons: [String] }`
  - `InProgressCommit { handle_id, est_commit_eta_ms: u32, queued_execute?: bool }`
  - `ConcurrencyCapReached { cap: u32, executing_count: u32, est_slot_eta_ms?: u32 }`
  - `VRAMFragmented { required_contiguous: u64, largest_free_block: u64, defrag_eta_ms?: u32 }`
  - `QuantizationTransformRequired { from: String, to: String, est_eta_ms: u32, supported: bool }`
  - `ResidentStaleVersion { handle_id, current_digest: String, requested_digest: String, est_recommit_eta_ms?: u32 }`
  - `MIGProfileMismatch { required_profile: String, have_profile: String }`
  - `TPGroupUnavailable { required_tp_degree: u32, available_devices: u32 }`
  - `AdmissionPaused { reason: String }`
  - `ThermalThrottling { est_recovery_ms?: u32, tps_degradation_pct?: u8 }`
  - `BatchJoinAdvised { active_batch_size: u32, est_speedup_factor: f32, est_slot_eta_ms?: u32 }`
  - `EvictionNotPermitted { reason: String }`
  - `DeferUntil { condition: { type: "AfterJobs", count: u32 } | { type: "TimeMs", ms: u32 } }`

Examples mapping (non-exhaustive):

- "I already have that model loaded in VRAM" → `UseResident { slots_free ≥ 0 }`
- "Loaded on VRAM but I'm inferring for someone else (x seconds)" → `UseResident { slots_free: 0, executing_count, executing_for_ms, est_slot_eta_ms }`
- "No room but I can evict something; queue is x" → `EvictThenCommitVRAM { shard_ids, bytes_to_free, est_eviction_eta_ms, est_commit_eta_ms }` (and/or `CannotFitVRAM { required, available, queue_depth }`)
- "I'm inferring (x seconds) but I can load it into VRAM now" → `CommitVRAM { est_commit_eta_ms, can_execute_now: false }` (commit in parallel if possible)
- "I'm inferring (x seconds) but I can evict then load" → `EvictThenCommitVRAM { ... }`
- "I can load and infer immediately" → `CommitVRAM { can_execute_now: true, est_commit_eta_ms: 0 }` (or `UseResident { slots_free > 0 }` if already resident)
- "I am already committing it; ETA ~300ms; you can queue now" → `InProgressCommit { est_commit_eta_ms: 300, queued_execute: true }`
- "Global slots exist, but this model's cap=2 is reached; wait ~2s" → `ConcurrencyCapReached { cap: 2, executing_count: 2, est_slot_eta_ms: 2000 }`
- "Total VRAM is enough, but contiguous block too small; defrag ~2s" → `VRAMFragmented { required_contiguous, largest_free_block, defrag_eta_ms: 2000 }`
- "Can quantize q5→q4_k_m to fit; ~1.2s; supported" → `QuantizationTransformRequired { from: "q5", to: "q4_k_m", est_eta_ms: 1200, supported: true }`
- "Resident digest differs; need re-commit ~800ms" → `ResidentStaleVersion { current_digest, requested_digest, est_recommit_eta_ms: 800 }`
- "Single-GPU worker; TP=2 required" → `TPGroupUnavailable { required_tp_degree: 2, available_devices: 1 }`
- "Thermals; expect -20% throughput for ~30s" → `ThermalThrottling { est_recovery_ms: 30000, tps_degradation_pct: 20 }`
- "Better to join current batch (size 3), ~1.5x speedup" → `BatchJoinAdvised { active_batch_size: 3, est_speedup_factor: 1.5 }`
- "Eviction needed but target is pinned" → `EvictionNotPermitted { reason: "Pinned handle" }`

---

## 3. Commit Endpoint

```
POST /worker/commit
```

- [WORKER-4220] The Commit endpoint MUST load model bytes into VRAM and seal the shard.
- [WORKER-4221] Request MUST include: `model_ref`, `shard_id`, `shard_index`, `model_bytes` (binary or path), `expected_digest`.
- [WORKER-4222] Workers MUST verify model signature before loading (if signature provided).
- [WORKER-4223] Workers MUST compute SHA-256 digest of model bytes and compare against `expected_digest` (if provided).
- [WORKER-4224] Workers MUST validate GGUF format defensively with bounds checking (max tensors, max file size).
- [WORKER-4225] Workers MUST fail fast if model bytes exceed `MAX_MODEL_SIZE` (configurable, default 100GB).
- [WORKER-4226] Response MUST include sealed `ModelShardHandle` with `sealed: true` and computed `digest`.
- [WORKER-4227] Workers MUST transition to `Ready` state only after successful commit and seal.

---

## 4. Ready Endpoint

```
GET /worker/ready
```

- [WORKER-4230] The Ready endpoint MUST attest that worker is ready with sealed shards.
- [WORKER-4231] Response MUST include: `ready: bool`, `handles: Vec<ModelShardHandle>`, `nccl_group_id: Option<String>`.
- [WORKER-4232] Workers MUST return `ready: false` if no model is loaded or seal verification fails.
- [WORKER-4233] The Ready endpoint MAY be unauthenticated for health checks (configurable).

---

## 5. Execute Endpoint

```
POST /worker/execute
```

- [WORKER-4240] The Execute endpoint MUST run inference with a sealed shard and stream tokens via SSE.
- [WORKER-4241] Request MUST include: `handle_id`, `prompt`, `params` (`max_tokens`, `temperature`, `seed`, etc.).
- [WORKER-4242] Workers MUST validate prompt length (max 100,000 chars by default, configurable).
- [WORKER-4243] Workers MUST validate `max_tokens` (max 4096 by default, configurable).
- [WORKER-4244] Workers MUST reject prompts containing null bytes (`\0`).
- [WORKER-4245] Workers MUST re-verify seal signature before execution.
- [WORKER-4246] Response MUST be SSE stream with events: `started`, `token`, `metrics`, `end`, `error`.
- [WORKER-4247] SSE `token` events MUST include: `{"t": "<token_text>", "i": <index>}`.
- [WORKER-4248] SSE `end` event MUST include: `{"tokens_out": <count>, "decode_time_ms": <duration>}`.

---

## 6. SSE Streaming Security

- [WORKER-4250] SSE streams MUST require authentication via Bearer token or job-specific token.
- [WORKER-4251] Workers MUST verify job ownership before streaming tokens (prevent cross-tenant leakage).
- [WORKER-4252] Workers MUST NOT emit tokens after cancellation (race-free cancel per ORCH-3026).
- [WORKER-4253] Workers MUST terminate streams after `event: error` or `event: end`; no further events MAY be sent.

---

## 7. Dependencies

**Crates used**:
- `vram-residency` — For ModelShardHandle and seal verification
- `model-loader` — For model validation before commit
- `capability-matcher` — For MCD/ECP checking in Plan
- `scheduler` — For job state tracking
- `input-validation` — For request validation
- `auth-min` — For Bearer token authentication

---

## 8. Traceability

**Code**: `bin/worker-orcd-crates/api/src/lib.rs`  
**Tests**: `bin/worker-orcd-crates/api/tests/`  
**Parent**: `bin/worker-orcd/.specs/00_worker-orcd.md` §3

---

## 9. Pool Control Endpoints (Capacity/Drain/Evict)

These endpoints enable pool-managerd to introspect VRAM capacity and control admission/VRAM residency on a per-worker basis. They complement Plan/Commit/Ready/Execute.

### 9.1 Capacity

```
GET /worker/capacity
```

- The Capacity endpoint MUST report current VRAM and slot capacity.
- Response MUST include: `worker_id`, `gpu_device`, `vram_total_bytes`, `vram_used_bytes`, `vram_free_bytes`, `slots_total`, `slots_free`, `draining`.
- Endpoint MUST require Bearer token authentication (same policy as other RPC endpoints).
- Endpoint SHOULD complete in < 10ms (query cached device list, fresh VRAM usage).

Example response:
```json
{
  "worker_id": "worker-gpu-0",
  "gpu_device": 0,
  "vram_total_bytes": 25769803776,
  "vram_used_bytes": 8589934592,
  "vram_free_bytes": 17179869184,
  "slots_total": 1,
  "slots_free": 1,
  "draining": false
}
```

### 9.2 Drain Control

```
POST /worker/drain
```

- Request body MUST include: `{ "drain": bool, "reason": String }` (reason MAY be empty).
- When `drain = true`, worker MUST close admission for Commit and Execute:
  - Plan MUST remain available (for planning and observability).
  - Commit MUST return `503 Service Unavailable` with a stable error code (e.g., `ADMISSION_CLOSED`).
  - Execute MUST return `503` unless the request refers to an in-flight job owned by the caller.
- When `drain = false`, worker MUST reopen admission.
- Endpoint MUST require Bearer token authentication.

Example request/response:
```json
{ "drain": true, "reason": "rolling-upgrade" }
```
```json
{ "draining": true }
```

### 9.3 Evict

```
POST /worker/evict
```

- Request body MUST include: `{ "shard_ids": [String] }`.
- Worker MUST attempt to unseal and free the specified shard(s) to reclaim VRAM.
- If a shard is currently executing, worker MUST return `409 Conflict` for that shard and leave it resident.
- Response MUST summarize outcomes: `{ "evicted": u32, "not_found": [String], "busy": [String] }`.
- Endpoint MUST require Bearer token authentication.

Example request/response:
```json
{ "shard_ids": ["shard-abc123", "shard-def456"] }
```
```json
{ "evicted": 2, "not_found": ["shard-def456"], "busy": [] }
```

### 9.4 Notes on Placement and Pinning

- Worker processes run per-GPU (or device mask) per `worker-orcd` spec §1.2; GPU selection is achieved by selecting the target worker, not by passing device indices to endpoints.
- Pool-managerd retains higher-level knowledge (including host RAM) and can use Capacity + Plan to build a sophisticated plan for `orchestratord` while the worker's Plan remains VRAM/capability-focused.

---

## 10. Refinement Opportunities

- Add `feasibility_only` flag to Plan to optionally skip MCD/ECP checks (still default on), if pool-managerd already vetted compatibility.
- Define stable error code catalog (e.g., `ADMISSION_CLOSED`, `VRAM_OOM`, `INVALID_PARAMS`) and document in this spec.
- Add mTLS support for internal RPC and callbacks (post-M0).
- Add SSE heartbeats/keepalives for long Execute streams (align with token streaming proposal 2025-09-19).
- Define rate limiting and backpressure behavior for Plan/Commit (protect VRAM operations).
- Consider job-scoped tokens for Execute (short-lived, signed), separate from worker token.

---

## 11. Batching Hints & Metrics

Batching is worker-internal and transparent to the HTTP API. This section standardizes optional hints and observability so orchestrators can reason about performance.

### 11.1 Optional Hints in Ready

- [WORKER-4260] `GET /worker/ready` Response MAY include per-handle metadata fields:
  - `supports_continuous_batching: bool`
  - `max_batch_size: u32`
  - `batch_window_ms: u32`
- [WORKER-4261] Absence of these fields implies default or unknown; clients MUST NOT assume batching is disabled.

### 11.2 Metrics (Prometheus)

- [WORKER-4265] Workers SHOULD expose batching metrics (names illustrative):
  - `worker_batch_size{handle_id}` (histogram)
  - `worker_batch_wait_ms{handle_id}` (histogram)
  - `worker_sequences_running{handle_id}` (gauge)
  - `worker_execute_started_total{handle_id}` (counter)
  - `worker_execute_completed_total{handle_id}` (counter)
  - `worker_execute_cancelled_total{handle_id}` (counter)
  - `worker_tokens_out_total{handle_id}` (counter)
  - `worker_decode_time_ms{handle_id}` (histogram)

### 11.3 Admission Semantics (Non-Breaking Clarifications)

- [WORKER-4268] While draining, new `Execute` admissions MUST return `503 ADMISSION_CLOSED`; in-flight batches MAY complete.
- [WORKER-4269] `POST /worker/execute` is per-handle; request MUST include `handle_id` (see §5 Execute).
