# SDK Types (Design Phase)

Status: draft (design-only)
Updated: 2025-09-21
Source of truth: `contracts/openapi/{control.yaml,data.yaml}`

Only fields that are (a) required by OpenAPI and (b) referenced by Utils are included.

## Core enums

- `Engine`: `"llamacpp" | "vllm" | "tgi" | "triton"` (OpenAPI `components.schemas.Engine`)
- `Workload`: `"completion" | "embedding" | "rerank"` (OpenAPI)
- `Priority`: `"interactive" | "batch"` (OpenAPI)
- `DeterminismLevel`: `"strict" | "best_effort"` (OpenAPI)
- `KVHint`: `"reuse" | "cold"` (OpenAPI)

## Discovery

- `EngineCapability`
  - `id: string`  // from Capabilities.engines[i].engine
  - `workloads: string[]`  // from Capabilities.engines[i].supported_workloads

- `Capabilities`
  - `api_version: string`
  - `engines: { engine: Engine, ctx_max: number, supported_workloads?: Workload[] }[]`

## Catalog

- `ModelInfo`
  - `id: string` (OpenAPI `CatalogModel.id`)
  - `digest: string` (OpenAPI `CatalogModel.digest`)
  - `state?: "Active" | "Retired"` (lifecycle; set via `POST /v1/catalog/models/{id}/state`)

## Pools (minimal)

- `PoolInfo`
  - `id: string`  // identifier only; OpenAPI does not define a list or GPU shape yet

## Data plane

- `TaskRequest` (subset aligned with OpenAPI `components.schemas.TaskRequest`)
  - `task_id: string` (uuid)
  - `session_id: string` (uuid)
  - `workload: Workload`
  - `model_ref: string`
  - `engine: Engine`
  - `ctx: number`
  - `priority: Priority`
  - `prompt?: string`
  - `inputs?: string[]`
  - `max_tokens: number`
  - `deadline_ms: number`
  - `seed?: number`
  - `determinism?: DeterminismLevel`
  - `sampler_profile_version?: string`
  - `expected_tokens?: number`
  - `kv_hint?: KVHint`
  - `placement?: PlacementOverrides`

- `PlacementOverrides`
  - `mode?: "pin" | "prefer" | "auto"`
  - `pin_pool_id?: string`
  - `prefer_pools?: string[]`
  - `avoid_pools?: string[]`
  - `require_device_mask?: string`
  - `allow_fallback?: boolean`

- `AdmissionResponse`
  - `task_id: string`
  - `queue_position: number`
  - `predicted_start_ms: number`
  - `backoff_ms: number`

- `SessionInfo`
  - `ttl_ms_remaining: number`
  - `turns: number`
  - `kv_bytes: number`
  - `kv_warmth: boolean`
  - `tokens_budget_remaining?: number`
  - `time_budget_remaining_ms?: number`
  - `cost_budget_remaining?: number`

## Errors

- `ErrorKind`: one of
  - `ADMISSION_REJECT | QUEUE_FULL_DROP_LRU | INVALID_PARAMS | POOL_UNREADY | POOL_UNAVAILABLE | REPLICA_EXHAUSTED | DECODE_TIMEOUT | WORKER_RESET | INTERNAL | DEADLINE_UNMET | MODEL_DEPRECATED | UNTRUSTED_ARTIFACT`

- `ErrorEnvelope`
  - `code: ErrorKind`
  - `message: string`
  - `engine?: Engine`
  - `retriable?: boolean`
  - `retry_after_ms?: number`
  - `policy_label?: string`  // required only for backpressure errors

## Streaming (SSE)

Event names and payloads per OpenAPI `data.yaml`:

- `SSEStarted`
  - `queue_position: number`
  - `predicted_start_ms: number`

- `SSEToken`
  - `t: string`  // token text
  - `i: number`  // incremental index

- `SSEMetrics` (additive fields; forward-compatible)
  - `on_time_probability?: number`
  - `queue_depth?: number`
  - `kv_warmth?: boolean`
  - `tokens_budget_remaining?: number`
  - `time_budget_remaining_ms?: number`
  - `cost_budget_remaining?: number`
  - `...additionalFields`

- `SSEEnd`
  - `tokens_out: number`
  - `decode_ms: number`

- `SSEError`
  - `code: ErrorKind`
  - `message: string`
  - `engine?: Engine`

- `SSEEvent` (union)
  - `{ name: "started", data: SSEStarted } | { name: "token", data: SSEToken } | { name: "metrics", data: SSEMetrics } | { name: "end", data: SSEEnd } | { name: "error", data: SSEError }`
