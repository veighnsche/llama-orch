## How to consume llama-orch

- Prefer **Utils** (`llama-orch-utils`) for end-to-end pipelines: applets, determinism helpers, proof bundles, and guardrails.
- Use the **SDK** (`llama-orch-sdk`) for low-level, typed access to HTTP/SSE and error envelopes; no applet or prompt logic lives here.
- Use the **CLI** (`llama-orch-cli`) to bootstrap and generate static capability/model/pool snapshots and language bindings; generated files import SDK types.
- The **Orchestrator** (`orchestratord`) API and OpenAPI contracts are the ground truth. The SDK mirrors them; Utils must not bypass the SDK.

# llama-orch — Consumer Capabilities and API Guide

This document is the consumer-facing, exhaustive guide to what llama-orch provides today: API contracts, behaviors, and capabilities you can rely on as a client. It consolidates the normative OpenAPI contracts under `contracts/openapi/` and the currently implemented HTTP/SSE routes in `orchestratord/`.

Authoritative contracts live in:
- `contracts/openapi/data.yaml` — Data plane (tasks, sessions, streaming)
- `contracts/openapi/control.yaml` — Control plane (capabilities, pools, catalog, artifacts)

If you observe any discrepancy between code and this guide, treat the OpenAPI contracts as the source of truth and file an issue.

## Audience & Scope

- API consumers building agents, CLIs, services that enqueue tasks and consume streamed tokens.
- Operators integrating pool control, catalog, and observability.
- Adapter authors mapping engine-native APIs to the llama‑orch adapter trait.

## Versioning & Compatibility

- Pre-1.0.0: no backwards-compatibility guarantees. Breaking changes may occur as we iterate. Always consult `GET /v1/capabilities` for `api_version` and feature discovery.
- Contracts are spec-first. When behavior evolves, OpenAPI specs update before runtime code.

## Operational Expectations

- GPU required; no CPU fallback. If no GPU capacity is available, the server will fail fast (e.g., `503 POOL_UNAVAILABLE`) or apply backpressure.
- VRAM-only residency: model weights, KV cache, and activations reside entirely in GPU VRAM during inference. No RAM↔VRAM sharing, unified memory/zero-copy, or host-RAM offload is supported. Tasks that do not fit must fail fast.
- Automatic provisioning: the program is responsible for preparing engines (e.g., llama.cpp, vLLM, TGI, Triton) and ensuring models per policy. Clients do not have to provision engines or models directly; enqueue requests may be accepted and streamed once the serving pool becomes ready.
- Single-host orchestration focus in current alpha. Multi-host or distributed scheduling may evolve later.

## Base URL, Authentication, and Headers

- Base URL: determined by the orchestrator process; typical local default is `http://127.0.0.1:8080/` when running `orchestratord`.
- Authentication:
  - API key: all endpoints except `/metrics` require header `X-API-Key: valid` (stubbed policy, will evolve).
  - Correlation: `X-Correlation-Id` is optional on requests; the server echoes it (or injects a UUID) in responses.
  - Worker registration (for adapters) uses Bearer token auth; see Worker Registration below.
- Backpressure headers on 429: `Retry-After` (seconds) and `X-Backoff-Ms` (milliseconds) when applicable.

## Capability Discovery

- `GET /v1/capabilities`
  - Returns a snapshot with `api_version` and engine families available. Example shape (see `orchestratord/src/services/capabilities.rs`):
    ```json
    {
      "api_version": "1.0.0",
      "engines": [
        {"engine": "llamacpp", "ctx_max": 32768, "supported_workloads": ["completion","embedding","rerank"], "rate_limits": {}, "features": {}},
        {"engine": "vllm",     "ctx_max": 32768, "supported_workloads": ["completion","embedding","rerank"], "rate_limits": {}, "features": {}},
        {"engine": "tgi",      "ctx_max": 32768, "supported_workloads": ["completion","embedding","rerank"], "rate_limits": {}, "features": {}},
        {"engine": "triton",   "ctx_max": 32768, "supported_workloads": ["completion","embedding","rerank"], "rate_limits": {}, "features": {}}
      ]
    }
    ```
  - Use this to gate features in your clients.

## Data Plane — Tasks, Streaming, Sessions

Contracts: `contracts/openapi/data.yaml`

### Enqueue a task

- `POST /v1/tasks`
- Body: `TaskRequest`
  - Required: `task_id (uuid)`, `session_id (uuid)`, `workload` (`completion|embedding|rerank`), `model_ref`, `engine` (`llamacpp|vllm|tgi|triton`), `ctx`, `priority` (`interactive|batch`), `max_tokens`, `deadline_ms`.
  - Optional: `seed`, `determinism` (`strict|best_effort`), `sampler_profile_version`, `prompt`, `inputs[]`, `expected_tokens`, `kv_hint` (`reuse|cold`), `placement` (see Placement below).
  - `model_ref` schemes (engine-dependent):
    - `hf:org/repo/path/file.gguf` (single-file GGUF; llama.cpp)
    - `hf:org/repo` (full repo; vLLM/TGI)
    - `file:/abs/path` or `relative/path`
    - `https://...`, `s3://bucket/key`, `oci://registry/repo:tag`
- Success: `202 Accepted` with body `AdmissionResponse { task_id, queue_position, predicted_start_ms, backoff_ms, streams?, preparation? }`
  - `streams.sse` and `streams.sse_verbose` provide direct URLs to the SSE endpoint; the latter is equivalent to `?verbose=true`.
  - `preparation.steps[]` (optional) announces tasks the server may perform before decode (e.g., `engine_provision`, `model_fetch`, `pool_warmup`). Use it to choose verbose streaming for richer progress UX.
  - Response headers may include budgets: `X-Budget-Tokens-Remaining`, `X-Budget-Time-Remaining-Ms`, `X-Budget-Cost-Remaining`.
- Error mapping (selected):
  - `400` `INVALID_PARAMS` (schema/sentinel violations)
  - `429` backpressure (`ADMISSION_REJECT` or `QUEUE_FULL_DROP_LRU`) with `Retry-After` and `X-Backoff-Ms`
  - `503` `POOL_UNAVAILABLE`
  - `500` `INTERNAL`

Example:
```bash
curl -s -H 'X-API-Key: valid' -H 'Content-Type: application/json' \
  -d '{
    "task_id":"00000000-0000-0000-0000-000000000001",
    "session_id":"11111111-1111-1111-1111-111111111111",
    "workload":"completion",
    "model_ref":"hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf",
    "engine":"llamacpp",
    "ctx":8192,
    "priority":"interactive",
    "prompt":"Hello",
    "max_tokens":64,
    "deadline_ms":30000
  }' \
  http://127.0.0.1:8080/v1/tasks | jq .
```

### Stream tokens and metrics (SSE)

- `GET /v1/tasks/{id}/stream` returns `text/event-stream`.
- Event sequence (normative): `started` → repeated `token` → optional repeated `metrics` → `end` (or `error`).
- Query parameter: `?verbose=true` — when set, the server may include human-narrated breadcrumbs and diagnostics within some `metrics` frames (e.g., `{ "human": "provisioning llama.cpp", "phase": "engine_provision" }`).
- Progress surface for CLI loading bars (optional): when available, `metrics` frames MAY include a `prep` object mapping step IDs to progress:
  ```json
  {
    "event": "metrics",
    "data": {
      "prep": {
        "engine:llamacpp": {"status":"running","bytes_done":10485760,"bytes_total":52428800,"pct":20,"phase":"compile","human":"Building llama.cpp (10/50 MB)"},
        "model:hf:org/repo/file.gguf": {"status":"running","bytes_done":73400320,"bytes_total":104857600,"pct":70,"phase":"download","human":"Downloading model (70/100 MB)"}
      }
    }
  }
  ```
  Clients can render concurrent bars by grouping step IDs by prefix (e.g., `engine:*`, `model:*`).
- Example frames (from `data.yaml`):
  ```text
  event: started
  data: {"queue_position":3,"predicted_start_ms":420}

  event: token
  data: {"t":"Hello","i":0}

  event: metrics
  data: {"queue_depth":2,"on_time_probability":0.82,"kv_warmth":true,
         "tokens_budget_remaining":12000,"time_budget_remaining_ms":45000,"cost_budget_remaining":0}

  event: end
  data: {"tokens_out":64,"decode_ms":2300}
  ```
- Response headers may include budgets at stream start (same `X-Budget-*` headers as enqueue).
- Optional micro-batching: set `ORCHD_SSE_MICROBATCH=1` on the server to merge consecutive `token` frames into `{"batch":[...]}` groups.
- Cancel-aware streaming: server periodically checks a cancel signal; see Cancel below.

Client tips:
- Use an SSE client with backpressure handling; do not assume newline boundaries align with `data:` payloads beyond the SSE spec.
- Treat fields in `metrics` frames as additive/forward-compatible.

### Cancel a task

- `POST /v1/tasks/{id}/cancel` → `204 No Content`.
- Semantics: client-initiated cancel should stop subsequent token emission. Current implementation emits no further tokens after cancel is observed during streaming.
- Recommended: if you also sever the SSE connection, still call explicit cancel — cancel-on-disconnect is a planned enhancement.

### Sessions

- `GET /v1/sessions/{id}` → `200` `SessionInfo { ttl_ms_remaining, turns, kv_bytes, kv_warmth, tokens_budget_remaining?, time_budget_remaining_ms?, cost_budget_remaining? }`
- `DELETE /v1/sessions/{id}` → `204` (evicts session/KV as applicable)
- Sessions are created lazily on first use and expire by TTL; every interaction may update turns/budgets.
 - Note: Sessions are metadata-only. No prompts, messages, or model outputs are stored in session state.
   Clients own conversation content. Session fields reflect TTL/turns/KV and budgets only.

## Control Plane — Pools, Catalog, Artifacts, Discovery

Contracts: `contracts/openapi/control.yaml`

### Pools (maintenance)

- `GET /v1/pools/{id}/health` → readiness, liveness, optional `metrics`, `draining` flag.
- `POST /v1/pools/{id}/drain` with `{"deadline_ms": <i64>}` → `202` starts draining.
- `POST /v1/pools/{id}/reload` with `{"new_model_ref": "..."}` → `202` on accept (may preload model/engine); sentinel inputs may yield `409`.

### Catalog (model registry)

- `POST /v1/catalog/models` — create/ingest a `CatalogModel` (id, digest, optional source/trust/attestations).
- `GET /v1/catalog/models/{id}` — retrieve catalog entry.
- `POST /v1/catalog/models/{id}/verify` — trigger verification; returns `202`.
- `POST /v1/catalog/models/{id}/state` — set lifecycle state (`Active|Retired`), optional `deadline_ms`; returns `202`.
- `DELETE /v1/catalog/models/{id}` — `204` on success, `404` if missing.

### Artifacts (plans, transcripts, traces)

- `POST /v1/artifacts` — body `Artifact { kind, content, tags?, parent?, metadata? }` → `201` `ArtifactRef { id, kind, ... }`.
- `GET /v1/artifacts/{id}` — `200` `Artifact` or `404`.
- Implementation detail: current IDs are `sha256:<hex>` of JSON content when using the default in-memory/fs stores.

### Worker Registration (adapters)

- `POST /v1/workers/register`
  - Auth: Bearer token required. The expected token is compared against `AUTH_TOKEN` server env (timing-safe equality).
  - Body (optional): `{ "pool_id": "default", "replica_id": "r0" }`
  - Response: `200` `{ ok: true, identity: "token:<fp>", pool_id, replica_id }`
  - Purpose: scaffolding to bind a worker adapter to a pool/replica during development.
  - Note: this endpoint is not yet part of the published OpenAPI contracts and is subject to change.

## Error Taxonomy & HTTP Mapping

The following error kinds appear in envelopes (`ErrorEnvelope` / `BackpressureErrorEnvelope`) and SSE `error` frames. Selected kinds:
- `ADMISSION_REJECT`
- `QUEUE_FULL_DROP_LRU`
- `INVALID_PARAMS`
- `POOL_UNREADY`
- `POOL_UNAVAILABLE`
- `REPLICA_EXHAUSTED`
- `DECODE_TIMEOUT`
- `WORKER_RESET`
- `INTERNAL`
- `DEADLINE_UNMET`
- `MODEL_DEPRECATED`
- `UNTRUSTED_ARTIFACT`

Mappings (typical):
- `400` → `INVALID_PARAMS`, `DEADLINE_UNMET`
- `429` → backpressure (`ADMISSION_REJECT` / `QUEUE_FULL_DROP_LRU`) with `Retry-After` and `X-Backoff-Ms`
- `503` → `POOL_UNAVAILABLE` (retriable)
- `500` → `INTERNAL`

Error envelope fields:
- `code`, `message`, optional `engine`, `retriable`, `retry_after_ms`, `policy_label` (for backpressure).

## Placement & GPU Selection

- Automatic placement: the orchestrator will choose an eligible pool/replica based on readiness/policy.
- API overrides via `TaskRequest.placement`:
  - `mode`: `pin | prefer | auto`
  - `pin_pool_id`: hard pin to a specific pool
  - `prefer_pools[]`: hint preferred pools
  - `avoid_pools[]`: avoid pools
  - `require_device_mask`: constrain device selection
  - `allow_fallback`: whether the system may fall back to other pools (default true)

Use overrides sparingly; they may reduce global efficiency.

## Observability

- `/metrics` (no API key required): Prometheus text; includes a correlation header. Key series include:
  - Counters: `tasks_enqueued_total`, `tasks_started_total`, `tasks_canceled_total{reason}`, `tasks_rejected_total{reason}`, `admission_backpressure_events_total{policy}`, `tokens_in_total`, `tokens_out_total`, `catalog_verifications_total`
  - Gauges: `queue_depth{...}`, `model_state{model_id,state}`, `kv_cache_usage_ratio{...}`, `gpu_utilization{...}`, `vram_used_bytes{...}`
  - Histograms: `latency_first_token_ms{...}`, `latency_decode_ms{...}`
- Logs: JSON lines with correlation. Common fields: `job_id`, `session_id`, `engine`, `engine_version`, `pool_id`, `replica_id`, `queue_position`, `predicted_start_ms`, `tokens_in`, `tokens_out`, `decode_time_ms`.
- SSE `metrics` frames: additive JSON surface for near-real-time client UX state.

### Narrative logging (human field)

- Purpose: provide a short, human-readable storyline alongside structured JSON logs so humans can skim flows without losing machine-readability.
- Shape: narration events attach the following fields (see `observability/narration-core/src/lib.rs::human()`):
  - `actor` — emitter (e.g., `"orchestratord"`).
  - `action` — verb or phase (e.g., `"admission"`, `"cancel"`, `"start"`).
  - `target` — primary subject/context (e.g., session ID, address, task ID).
  - `human` — short English sentence (≤ ~100 chars) describing intent/outcome.
- Where emitted today:
  - Startup and HTTP/2 preference notes (see `orchestratord/src/app/bootstrap.rs`).
  - Task admission (enqueue) breadcrumbs (see `orchestratord/src/api/data.rs::create_task`).
  - Task cancel requests (see `orchestratord/src/api/data.rs::cancel_task`).
  - Additional touchpoints (placement, stream start/end) are planned per `.specs/proposals/2025-09-19-human-narration-logging.md`.
- Output format: logs are emitted via `tracing_subscriber` in JSON by default (`init_observability()`); pretty console mode is planned but not required for consumers. Control verbosity with `RUST_LOG`.
- Security & style: narration MUST be natural language (human-friendly) and MUST NOT consist primarily of opaque IDs; do not include secrets/PII; keep present tense, subject‑verb‑object; taxonomy complements structured fields (`README_LLM.md`). Keep raw IDs in structured fields (e.g., `job_id`, `session_id`) rather than in `human`.
- Example JSON log (illustrative):

  ```json
  {
    "timestamp": "2025-09-21T11:20:31.000Z",
    "level": "INFO",
    "actor": "orchestratord",
    "action": "admission",
    "target": "pool:default",
    "human": "Accepted request; queued at position 3 (ETA 420 ms) on pool 'default'",
    "job_id": "00000000-0000-0000-0000-000000000001",
    "session_id": "11111111-1111-1111-1111-111111111111",
    "engine": "llamacpp",
    "pool_id": "default",
    "queue_position": 3,
    "predicted_start_ms": 420,
    "x_correlation_id": "11111111-1111-4111-8111-111111111111"
  }
  ```
- Consumer guidance:
  - Treat `human` as additive context; rely on structured fields for automation.
  - Filter JSON logs on `actor`, `action`, and `target` to follow a story; correlate with HTTP via `X-Correlation-Id` headers.
  - Expect future adoption across more events; the presence of the `human` field is repo-wide policy per the accepted proposal.

## Determinism & Reproducibility

- Deterministic default paths for SSE emission in dev mode; transcripts are persisted as artifacts for replay/validation.
- Seeds and sampler profiles are plumbed in `TaskRequest` for deterministic generation where supported by engines.

## Client Libraries and Tooling

- Use the OpenAPI specs under `contracts/openapi/` to generate clients.
- The workspace includes a sample OpenAPI client tool under `tools/openapi-client/`.

## End-to-End Example (local)

```bash
# Capabilities
curl -s -H 'X-API-Key: valid' http://127.0.0.1:8080/v1/capabilities | jq .

# Enqueue a task
curl -s -H 'X-API-Key: valid' -H 'Content-Type: application/json' \
  -d '{"task_id":"t1","session_id":"s1","workload":"completion","model_ref":"hf:org/repo/file.gguf","engine":"llamacpp","ctx":8192,"priority":"interactive","prompt":"Hello","max_tokens":8,"deadline_ms":60000}' \
  http://127.0.0.1:8080/v1/tasks | jq .

# Stream SSE
curl -s -H 'X-API-Key: valid' http://127.0.0.1:8080/v1/tasks/t1/stream

# Cancel
curl -s -X POST -H 'X-API-Key: valid' http://127.0.0.1:8080/v1/tasks/t1/cancel -i

# Session
curl -s -H 'X-API-Key: valid' http://127.0.0.1:8080/v1/sessions/s1 | jq .

# Artifact
curl -s -H 'X-API-Key: valid' -H 'Content-Type: application/json' \
  -d '{"kind":"trace","content":{"events":[{"type":"started","data":{}}]}}' \
  http://127.0.0.1:8080/v1/artifacts | jq .
```

## Known Limitations & Roadmap Notes

- Multi-token adapter-driven streaming exists but is scaffolded; end-to-end real engine streaming is under active development.
- Cancel-on-disconnect: planned; explicit `POST /cancel` is recommended today.
- Capability discovery is static today; dynamic pool/adapter introspection is on the roadmap.
- Placement policy, rate limits, and per-pool quotas are evolving; treat `rate_limits` and `features` as hints.

## Appendix — Type Summaries

- `TaskRequest` (see Data Plane OpenAPI): identifies job, session, workload, model, engine, and decode controls. Includes placement overrides.
- `AdmissionResponse`: `{ task_id, queue_position, predicted_start_ms, backoff_ms, streams?, preparation? }`.
  - `streams`: `{ sse: uri, sse_verbose: uri }`
  - `preparation`: `{ steps: [ { kind: 'engine_provision'|'model_fetch'|'pool_warmup', description?, estimated_ms? } ] }`
- SSE events: `SSEStarted`, `SSEToken`, `SSEMetrics`, `SSEEnd`, `SSEError`.
- `SessionInfo`: TTL/turns/KV and optional budgets.
- `CatalogModel`: id, digest, optional source/trust/manifests/signatures/SBOM.
- `Artifact` and `ArtifactRef`: flexible kinds and metadata for auxiliary documents.

---

Feedback welcome. If anything is unclear for your integration, please open an issue with your use case and an example payload/trace.
