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
