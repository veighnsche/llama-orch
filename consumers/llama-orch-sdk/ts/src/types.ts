// Design-phase stubs — types only. No runtime logic.
// Mirrors consumers/llama-orch-sdk/.docs/02-types.md

export type Engine = "llamacpp" | "vllm" | "tgi" | "triton";
export type Workload = "completion" | "embedding" | "rerank";
export type Priority = "interactive" | "batch";
export type DeterminismLevel = "strict" | "best_effort";
export type KVHint = "reuse" | "cold";

export interface EngineCapability {
  id: string;               // from Capabilities.engines[i].engine
  workloads: Workload[];    // from Capabilities.engines[i].supported_workloads
}

export interface CapabilitiesEngineEntry {
  engine: Engine;
  ctx_max: number;
  supported_workloads?: Workload[];
}

export interface Capabilities {
  api_version: string;
  engines: CapabilitiesEngineEntry[];
}

export interface ModelInfo {
  id: string;
  digest: string;
  state?: "Active" | "Retired";
}

export interface PoolInfo {
  id: string;
}

export type PlacementMode = "pin" | "prefer" | "auto";

export interface PlacementOverrides {
  mode?: PlacementMode;
  pin_pool_id?: string;
  prefer_pools?: string[];
  avoid_pools?: string[];
  require_device_mask?: string;
  allow_fallback?: boolean;
}

export interface TaskRequest {
  task_id: string; // uuid
  session_id: string; // uuid
  workload: Workload;
  model_ref: string;
  engine: Engine;
  ctx: number;
  priority: Priority;
  prompt?: string;
  inputs?: string[];
  max_tokens: number;
  deadline_ms: number;
  seed?: number;
  determinism?: DeterminismLevel;
  sampler_profile_version?: string;
  expected_tokens?: number;
  kv_hint?: KVHint;
  placement?: PlacementOverrides;
}

export interface AdmissionResponse {
  task_id: string;
  queue_position: number;
  predicted_start_ms: number;
  backoff_ms: number;
}

export interface SessionInfo {
  ttl_ms_remaining: number;
  turns: number;
  kv_bytes: number;
  kv_warmth: boolean;
  tokens_budget_remaining?: number;
  time_budget_remaining_ms?: number;
  cost_budget_remaining?: number;
}

export type ErrorKind =
  | "ADMISSION_REJECT"
  | "QUEUE_FULL_DROP_LRU"
  | "INVALID_PARAMS"
  | "POOL_UNREADY"
  | "POOL_UNAVAILABLE"
  | "REPLICA_EXHAUSTED"
  | "DECODE_TIMEOUT"
  | "WORKER_RESET"
  | "INTERNAL"
  | "DEADLINE_UNMET"
  | "MODEL_DEPRECATED"
  | "UNTRUSTED_ARTIFACT";

export interface ErrorEnvelope {
  code: ErrorKind;
  message: string;
  engine?: Engine;
  retriable?: boolean;
  retry_after_ms?: number;
  policy_label?: string; // backpressure-only
}

export interface SSEStarted { queue_position: number; predicted_start_ms: number; }
export interface SSEToken { t: string; i: number; }
export interface SSEMetrics {
  on_time_probability?: number;
  queue_depth?: number;
  kv_warmth?: boolean;
  tokens_budget_remaining?: number;
  time_budget_remaining_ms?: number;
  cost_budget_remaining?: number;
  // additional fields may appear; treat as forward-compatible
  [k: string]: unknown;
}
export interface SSEEnd { tokens_out: number; decode_ms: number; }
export interface SSEError { code: ErrorKind; message: string; engine?: Engine; }

export type SSEEvent =
  | { name: "started"; data: SSEStarted }
  | { name: "token"; data: SSEToken }
  | { name: "metrics"; data: SSEMetrics }
  | { name: "end"; data: SSEEnd }
  | { name: "error"; data: SSEError };
