# Proposal: Close Home‑Profile Gaps for Client Simplicity (SPEC CHANGE)

Status: Draft

Owner: @llama-orch-maintainers

Date: 2025-09-20

## 0) Motivation
Small, additive clarifications reduce client error rates and simplify tooling. A uniform SSE error frame, richer capability discovery for pre‑validation, minimal per‑job artifacts, and an optional output‑mode hint help clients behave predictably without guessing. All changes are backward compatible and align with the Home Profile goals.

## 1) Scope
In scope:
- Data‑plane streaming (SSE) error semantics and documentation of client classes.
- Capability discovery payload completeness for pre‑validation.
- Minimal artifacts/proof bundles per job (including failure cases).
- Optional enqueue hint for output packaging.
- Optional CORS/preflight for localhost tooling.
- Server expectations for budgets/capabilities and optional tokenization.

Out of scope:
- Removals/deprecations; breaking API changes.
- Engine‑specific internals or non‑Home Profile behaviors.

## 2) Normative Requirements (RFC‑2119)
IDs are Draft; maintainers may renumber on merge. Where possible, IDs follow existing ranges used in `/.specs/00_llama-orch.md`.

### 2.1 SSE Error Frame (Uniform)
- [ORCH‑3406] When a streaming failure occurs after stream start, the server MUST emit `event: error` with a minimal JSON body: `{ code: string, retriable: boolean, retry_after_ms?: number, message?: string }`.
- [ORCH‑3407] After `event: error`, the stream MUST terminate; no further `token`/`metrics` events.
- [ORCH‑3408] Pre‑stream errors MUST use HTTP status codes; established streams MUST stay `200` and carry the error via the SSE `error` event.
- [ORCH‑3409] `code` MUST map to `ErrorKind` (contracts). `SSEError` in `contracts/openapi/data.yaml` SHOULD be extended additively to include `retriable` and `retry_after_ms?`.

### 2.2 Capabilities Payload Completeness (Client Pre‑Validation)
- [ORCH‑3093] `GET /v1/capabilities` MUST include per engine/pool: `engine`, `engine_version`, `sampler_profile_version`, `ctx_max`, `max_tokens_out`, declared concurrency/slots, and `supported_workloads`.
- [ORCH‑3094] Clients SHOULD pre‑validate context/output/sampler compatibility before enqueue; the server MUST still enforce guardrails (ORCH‑3014) and return structured errors when violated.
- [ORCH‑3096] Capability payloads MUST continue to include `api_version` compatible with OpenAPI `info.version`.

### 2.3 Artifacts / Proof Bundles (Per‑Job)
- [ORCH‑3350] Each job SHOULD produce an artifact record even on failure, referencing: `job_id`, `session_id`, request params (prompt/inputs, `max_tokens`, `ctx`, `engine`, `engine_version`, `sampler_profile_version`, `model_ref`, digest when known), `seed?`, key metrics (`tokens_out`, `decode_time_ms`), and an SSE transcript (inline or by reference).
- [ORCH‑3351] If an HTTP artifact registry exists, it SHOULD follow `POST /v1/artifacts` and `GET /v1/artifacts/{id}` (as in contracts). If not, the local storage layout and naming MUST be documented.
- [ORCH‑3352] Failure paths SHOULD still create an artifact with error context (`code`, `message`, `retriable`, `retry_after_ms?`) and partial transcript when available.

### 2.4 Output Mode Hint (Optional)
- [ORCH‑3101] `TaskRequest` MAY include `output_mode: "text" | "json" | "edits"` as a hint for packaging/validation and artifact tagging. Unknown values MUST be ignored.
- [ORCH‑3102] Validation remains client‑side; servers MUST NOT change model semantics based on this hint.

### 2.5 Error‑Class Documentation (Client Behavior)
- [ORCH‑3330] Codify classes:
  - Reject upfront → HTTP 400 (e.g., `INVALID_PARAMS`, `DEADLINE_UNMET`).
  - Retry later → HTTP 429 with `Retry-After` and `X-Backoff-Ms` (e.g., `ADMISSION_REJECT`, `QUEUE_FULL_DROP_LRU`).
  - Permanent fail → HTTP 5xx or SSE `event:error` with `retriable=false` (e.g., `POOL_UNAVAILABLE`, `WORKER_RESET`, `INTERNAL`).
- [ORCH‑3331] Maintain a short mapping aligned to `ErrorKind` and SSE `error` frames.

### 2.6 CORS / Preflight (MAY)
- [ORCH‑3380] Implementations MAY support CORS for localhost tooling, replying to `OPTIONS` with `Access-Control-Allow-*` headers (include `X-Correlation-Id`, and `Authorization` when Minimal Auth seam is active).
- [ORCH‑3381] Disabled by default; enabling MUST be non‑breaking.

## 3) API Sketches (Non‑Normative)
Capabilities (extended):
```json
{ "api_version": "1.0.0", "engines": [ { "engine": "llamacpp", "engine_version": "b1234", "sampler_profile_version": "v1", "ctx_max": 32768, "max_tokens_out": 4096, "concurrency": 1, "supported_workloads": ["completion"] } ] }
```
Enqueue with hint:
```json
{ "task_id": "...", "session_id": "...", "workload": "completion", "model_ref": "...", "engine": "llamacpp", "ctx": 8192, "priority": "interactive", "max_tokens": 64, "deadline_ms": 30000, "output_mode": "text" }
```
SSE error example:
```
event: started
data: {"queue_position":3,"predicted_start_ms":420}

event: error
data: {"code":"DECODE_TIMEOUT","retriable":true,"retry_after_ms":1000,"message":"decode exceeded deadline"}
```

## 4) Server Expectations
The server MUST:
- Expose model capabilities: `ctx_max`, `max_tokens_out`, and MAY include `tokenizer_name`. (Aligns with [ORCH‑3093].)
- Reject over‑budget requests cleanly: fail fast with 400/429 style errors (no silent truncation). (Extends ORCH‑3014/ORCH‑2007) → [ORCH‑3016] No silent truncation permitted; enforce limits before enqueue or at admission with structured errors.

The server SHOULD:
- Report `tokens_used` and `tokens_remaining` for active sessions (e.g., in the `started` SSE frame or the first `metrics` frame). → [ORCH‑3105] Additive; when unavailable, omit fields.
- Optionally expose `GET /v1/tokenize` (or equivalent) for exact counts by model. → [ORCH‑3390] MAY be implemented; purely advisory and unauthenticated in Home Profile.
- Note: This is opt‑in. The CLI MUST NOT call it automatically; repos choose to use it. → [ORCH‑3391]

## 5) Backward Compatibility
- SSE `error` fields are additive; absence implies legacy behavior.
- Capabilities fields are additive; missing MAY be treated as unknown.
- `output_mode` is optional input.
- Artifacts are additive; absence preserves current behavior.
- Server expectations add clarity; over‑budget rejection already aligns with existing 400/429 taxonomy.

## 6) Testing Implications
- Unit/integration: SSE `error` emission/termination; capabilities fields present/coherent; `output_mode` echoed/tagged in artifacts; error‑class mapping in sync; artifact creation on failure.
- Add tests for no silent truncation on over‑budget inputs; assert 400/429 with envelopes and headers.
- If `/v1/tokenize` is implemented, add happy‑path and edge‑case tests; ensure CLI does not rely on it implicitly.

## 7) Operational Notes
- Narration/logging MUST NOT leak secrets in error or SSE frames (align with narration spec ORCH‑33xx). Keep metric names aligned with `README_LLM.md` and `.specs/metrics/otel-prom.md`.

## 8) Rollout Plan
- Additive changes; safe to deploy. If feature flags exist, gate SSE error extras, capabilities enrichment, and optional endpoints.

## 9) Open Questions
- Exact capability field names per engine/pool (e.g., `concurrency` vs `slots_total`) — confirm in orchestrator/adapters.
- Artifact backend defaults — HTTP vs. documented filesystem layout.
- Policy for `retriable`/`retry_after_ms` mapping across engines on SSE `error`.
- Where to surface `tokens_used`/`tokens_remaining` (headers vs. `started` vs. `metrics`) as a stable convention.

## 10) Refinement Opportunities
- Expand capabilities with sampler/profile compatibility matrices as adapters expose them.
- Minimal artifact list/browse endpoint (tag/prefix) if client use cases emerge (non‑breaking).
- Tokenization endpoint shape and caching guidance; tokenizer name normalization.

## 11) Prior Art / Alignment
- Root spec: `/.specs/00_llama-orch.md` (§2.5 Streaming & Determinism, §2.6 Catalog/Artifacts, §2.9 Capability Discovery)
- Proposals: `/.specs/proposals/2025-09-19-token-streaming-and-cancel-robustness.md`, `/.specs/proposals/2025-09-19-human-narration-logging.md`
- Contracts: `contracts/openapi/data.yaml`, `contracts/openapi/control.yaml`
- Process: `/.docs/PROCESS.md`
