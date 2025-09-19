# Proposal: Nerve‑Native Server API and Core Runtime

Status: Draft

Owner: @llama-orch-maintainers

Date: 2025-09-19

## 0) Motivation

Serve Nerve as the native execution model. Replace legacy orchestrator/agent/worker surfaces with a minimal, LLM‑first contract that is deterministic, policy‑enforced, and placement‑aware.

## 1) Scope

In scope:
- Core objects: `Flow`, `Run`, `Step`, `Pool`, `Fact`, `Checkpoint`.
- Submit plans (`llm_plan`, `llm_call`, `tool_call`, `fs_op`, `gate_eval`, `checkpoint`).
- Typed streaming of events/facts; budgets, seeds, policies enforced server‑side.
- First‑class pools/placement: `auto`, `pin(pool_id)`, `affinity(tags)`.
- Deterministic replay with provenance.

Out of scope: Backward compatibility with current endpoints.

## 2) Normative Requirements (RFC‑2119)

IDs: NERVE‑4xxx (core), STREAM‑2xxx (stream), POLICY‑3xxx (policy), PLACE‑5xxx (placement).

### Core Model
- [NERVE‑4000] Server MUST expose first‑class: `Flow`, `Run`, `Step`, `Pool`, `Fact`, `Checkpoint` with stable IDs.
- [NERVE‑4001] `Flow` MUST be content‑addressed (hash of normalized Nerve AST + tool schemas + config). Mutations MUST create a new `Flow`.
- [NERVE‑4002] `Run` MUST carry `mode` (`record|replay|strict_replay`), `seed`, budgets, policy, pool policy, and FS policy.
- [NERVE‑4003] `Step` types: `llm_plan`, `llm_call`, `tool_call`, `fs_op`, `gate_eval`, `checkpoint`. Steps MUST be idempotent within a `Run` by `client_token`.
- [NERVE‑4004] `Fact` MUST be append‑only, typed, schema‑validated, queryable by `Run`/`Flow`.
- [NERVE‑4005] `Checkpoint` MUST snapshot interpreter‑relevant state for replay across restarts.

### LLM Semantics & Validation
- [NERVE‑4010] LLM steps MUST support `plan` (produce structured intents/tool suggestions) and `call` (execute) modes.
- [NERVE‑4011] Tool menus MUST be structured JSON referencing registered tool schemas (JSON Schema 2020‑12). Server MUST validate tool args.
- [NERVE‑4012] Prompts/params MUST be normalized and hashed; `seed` MUST be logged/enforced.

### Streaming & Events
- [STREAM‑2000] Server MUST stream typed events over SSE incrementally with bounded memory; flush per event.
- [STREAM‑2001] Per step, event order MUST be `started → [token|tool_suggest|fact|metrics|narration]* → [checkpoint]? → end` (or `error`). No tokens after cancel.
- [STREAM‑2002] A per‑step `CancellationToken` MUST be plumbed end‑to‑end. Client disconnect SHOULD trigger cancel (configurable).
- [STREAM‑2003] Backpressure MUST be bounded; `metrics`/`narration` MAY drop first; `token`/`fact` MUST NOT drop. Optional `: keep-alive` heartbeats MAY be emitted.

### Determinism & Provenance
- [NERVE‑4020] Every request/event MUST log: `seed`, `prompt_hash`, `params_hash`, `engine`, `model_hash`, `pool_id`, `replica_id`, `tokens_in`, `tokens_out`, `cost_usd`, `decode_time_ms`.
- [NERVE‑4021] `replay` MUST reproduce prior outputs; `strict_replay` MUST match byte‑exact token streams.
- [NERVE‑4022] `strict_replay` MUST be rejected if determinism inputs differ (prompt/params/seed/engine/model/replica/policy material to output).

### Policy, Budgets, Safety
- [POLICY‑3000] Hard budgets MUST be enforced at `Run` and `Step`: `max_tokens_in`, `max_tokens_out`, `max_wall_ms`, `max_cost_usd`.
- [POLICY‑3001] FS broker MUST enforce sandbox roots, allow‑lists, quotas; violations MUST error and MAY trip a circuit breaker.
- [POLICY‑3002] Circuit breaker MUST guard external calls and emit `escalation` events.
- [POLICY‑3003] Server‑side policy evaluation (allow/deny, redact) MUST occur prior to execution.

### Pools & Placement
- [PLACE‑5000] Pools MUST be first‑class with discoverable capabilities (VRAM, arch, features, engines, load).
- [PLACE‑5001] Placement MUST support `auto`, `pin(pool_id)`, `affinity(tags/selector)` with deterministic tie‑break in `strict_replay`.
- [PLACE‑5002] Server SHOULD provision engines automatically when policy allows; pinning MUST be honored.

## 3) Core Objects & Lifecycle (shape)

- Flow: `flow_id`, `hash`, `ast_ir`, `tools[]`, `labels`, `created_at`.
  - Create: `POST /v1/flows`; Get: `GET /v1/flows/{id}`; By hash: `GET /v1/flows:by-hash/{hash}`.
- Run: `run_id`, `flow_id`, `mode`, `seed`, `budgets`, `policy`, `pool_policy`, `fs_policy`, `status`, `provenance`.
  - Create: `POST /v1/flows/{id}/runs`; Get: `GET /v1/runs/{id}`; Cancel: `POST /v1/runs/{id}/cancel`.
- Step: `step_id`, `run_id`, `type`, `client_token`, `input`, `placement`, `status`, `budget_usage`, `provenance`.
  - Submit: `POST /v1/runs/{id}/steps`; Get: `GET /v1/steps/{id}`; Stream: `GET /v1/steps/{id}/stream`; Cancel: `POST /v1/steps/{id}/cancel`.
- Fact: `fact_id`, `run_id`, `type`, `schema_uri`, `data`, `emitted_by_step`, `ts`.
  - Query: `GET /v1/runs/{id}/facts`; Emit: `POST /v1/runs/{id}/facts` (interpreter facts).
- Checkpoint: `checkpoint_id`, `run_id`, `step_id?`, `snapshot`, `ts`.
  - Create: `POST /v1/runs/{id}/checkpoint`; List: `GET /v1/runs/{id}/checkpoints`.
- Pool: `pool_id`, `tags`, `capabilities`, `engines`, `replicas(health/load)`.
  - List: `GET /v1/pools`; Details: `GET /v1/pools/{id}`; Provision: `POST /v1/pools/provision` (optional).

## 4) API Endpoints (selected)

Common headers: accepts/echoes `X-Correlation-Id`. JSON MUST include IDs and status.

- POST `/v1/flows` → `{ flow_id, hash }`.
- POST `/v1/flows/{flow_id}/runs` → `{ run_id, status }` including budgets, policy, pool_policy, fs_policy, seed, mode.
- POST `/v1/runs/{run_id}/steps` → `{ step_id, status }` with union `input` by `type`.
- GET `/v1/steps/{step_id}/stream` → SSE events: `started`, `token`, `tool_suggest`, `fact`, `metrics`, `narration`, `checkpoint`, `end`, `error`, `escalation`.
- POST `/v1/steps/{step_id}/cancel` → `{ status: "cancelling" }`.
- GET `/v1/runs/{run_id}/facts` → array of typed facts.
- GET `/v1/pools` / `/v1/pools/{id}` → capabilities, engines, load; `POST /v1/pools/provision` optional.

## 5) Event / Streaming Model

Transport: SSE (`text/event-stream`). Headers: `Cache-Control: no-cache`, `Connection: keep-alive`, `X-Accel-Buffering: no` (advisory).
- Frame ordering per [STREAM‑2001].
- Bounded mpsc channel; drop policy per [STREAM‑2003].
- Optional micro‑batching (disabled by default). Optional `: keep-alive` heartbeats.
- Narration frames rate‑limited; never per‑token logs.

## 6) Interpreter → API Mapping

- flow compile → `POST /v1/flows`.
- start run → `POST /v1/flows/{flow_id}/runs` (mode/seed/budgets/policies/placement/fs_policy).
- for each Nerve `step` → `POST /v1/runs/{run_id}/steps`; then `GET /v1/steps/{step_id}/stream` until terminal.
- emit fact from interpreter → `POST /v1/runs/{run_id}/facts`.
- cancel step/run → `POST /v1/steps/{id}/cancel` or `POST /v1/runs/{id}/cancel`.
- gate wait → poll/query facts or use SSE facts stream.
- placement → set `placement` field (`auto|pin|affinity`).

## 7) Destructive Changes vs Old Design

- Drop legacy `/tasks`, `/admission`, `/queue` routes; replace with `flows/runs/steps`.
- Collapse “agent/job/task” into `Step`.
- Make pools/placement explicit; remove implicit engine routing.
- Replace ad‑hoc streaming with typed SSE frames and determinism metadata.
- Server‑side tool schema registry and validation (no client‑side only validation).

## 8) Spec Updates To Apply

- `/.specs/00_llama-orch.md` and `20_orchestratord.md`: replace HTTP surface with `flows/runs/steps`, SSE model, budgets/policies, placement.
- `/.specs/metrics/otel-prom.md`: add determinism/provenance fields and backpressure counters.
- `worker-adapters/` specs: adopt tool arg validation via server schemas; ensure cancel and deterministic streaming.

## 9) CI & Testing

- Unit: event order, budgets enforcement, idempotency by `client_token`, replay/strict_replay.
- Integration: cancel-on-disconnect, FS broker allow‑lists/quotas, pool pin/affinity.
- Determinism: byte‑exact streams under `strict_replay` on same replica.

## 10) Refinement Opportunities

- gRPC bi‑di stream for advanced tool control (future toggle).
- Materialized fact indices for low‑latency gates.
- Adaptive micro‑batching based on RTT/backpressure.
- Policy plugin hooks (OPA/Rego or WASM) with caching.

## 11) Mapping to Repo Reality (Anchors)

- `orchestratord/` → HTTP routes for `flows/runs/steps`, SSE emitter, policy/budget middleware, FS broker.
- `orchestrator-core/` → queue/placement, idempotency, determinism/provenance, budgets.
- `worker-adapters/adapter-api` + `adapter-host/` → LLM call streaming, cancel propagation, placement bindings.
- `contracts/api-types/` → new objects and enums; `contracts/openapi/` → new surface.
- `tools/openapi-client/` → regenerate; `test-harness/` → BDD for replay/strict_replay and policy.

## 12) Open Questions

- Transport alternatives: Should we standardize on SSE only or keep an experimental gRPC stream behind a feature flag for non-browser clients?
- Fact store: Do we require indexing and query DSL now (e.g., SQLite/JSON1 or Tantivy) or start with append-only logs plus simple filters?
- Tool sandboxing: Is FS broker sufficient or do we need per-tool process isolation via WASM/OCI for stronger guarantees in Home Profile?
- Cost accounting: Which provider price tables are authoritative and how do we version them for reproducible `strict_replay` cost fields?
- Engine provisioning: How far should auto-provision go (download, compile, cache) in the default profile without prompting the user?
