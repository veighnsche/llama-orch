// Design-phase stub — types only; no logic.
// Mirrors consumers/llama-orch-sdk/.docs/02-types.md

use serde::{Deserialize, Serialize};
#[cfg(feature = "wasm")]
use tsify::Tsify;

#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Engine {
    Llamacpp,
    Vllm,
    Tgi,
    Triton,
}

#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Workload {
    Completion,
    Embedding,
    Rerank,
}

#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Priority {
    Interactive,
    Batch,
}

#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DeterminismLevel {
    Strict,
    BestEffort,
}

#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum KVHint {
    Reuse,
    Cold,
}

#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct EngineCapability {
    pub id: String,
    pub workloads: Vec<Workload>,
}

#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct CapabilitiesEngineEntry {
    pub engine: Engine,
    pub ctx_max: i32,
    pub supported_workloads: Option<Vec<Workload>>,
}

#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Capabilities {
    pub api_version: String,
    pub engines: Vec<CapabilitiesEngineEntry>,
}

#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ModelInfo {
    pub id: String,
    pub digest: String,
    pub state: Option<ModelState>,
}

#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelState {
    Active,
    Retired,
}

#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct PoolInfo {
    pub id: String,
}

#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum PlacementMode {
    Pin,
    Prefer,
    Auto,
}

#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct PlacementOverrides {
    pub mode: Option<PlacementMode>,
    pub pin_pool_id: Option<String>,
    pub prefer_pools: Option<Vec<String>>,
    pub avoid_pools: Option<Vec<String>>,
    pub require_device_mask: Option<String>,
    pub allow_fallback: Option<bool>,
}

#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct TaskRequest {
    pub task_id: String,
    pub session_id: String,
    pub workload: Workload,
    pub model_ref: String,
    pub engine: Engine,
    pub ctx: i32,
    pub priority: Priority,
    pub prompt: Option<String>,
    pub inputs: Option<Vec<String>>,
    pub max_tokens: i32,
    pub deadline_ms: i64,
    pub seed: Option<i64>,
    pub determinism: Option<DeterminismLevel>,
    pub sampler_profile_version: Option<String>,
    pub expected_tokens: Option<i32>,
    pub kv_hint: Option<KVHint>,
    pub placement: Option<PlacementOverrides>,
}

#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct AdmissionResponse {
    pub task_id: String,
    pub queue_position: i32,
    pub predicted_start_ms: i64,
    pub backoff_ms: i64,
}

#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SessionInfo {
    pub ttl_ms_remaining: i64,
    pub turns: i32,
    pub kv_bytes: i64,
    pub kv_warmth: bool,
    pub tokens_budget_remaining: Option<i64>,
    pub time_budget_remaining_ms: Option<i64>,
    pub cost_budget_remaining: Option<f32>,
}

#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ErrorKind {
    AdmissionReject,
    QueueFullDropLru,
    InvalidParams,
    PoolUnready,
    PoolUnavailable,
    ReplicaExhausted,
    DecodeTimeout,
    WorkerReset,
    Internal,
    DeadlineUnmet,
    ModelDeprecated,
    UntrustedArtifact,
}

#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct ErrorEnvelope {
    pub code: ErrorKind,
    pub message: String,
    pub engine: Option<Engine>,
    pub retriable: Option<bool>,
    pub retry_after_ms: Option<i64>,
    pub policy_label: Option<String>,
}

#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SSEStarted {
    pub queue_position: i32,
    pub predicted_start_ms: i64,
}

#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SSEToken {
    pub t: String,
    pub i: i32,
}

#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct SSEMetrics {
    pub on_time_probability: Option<f32>,
    pub queue_depth: Option<i32>,
    pub kv_warmth: Option<bool>,
    pub tokens_budget_remaining: Option<i64>,
    pub time_budget_remaining_ms: Option<i64>,
    pub cost_budget_remaining: Option<f32>,
}

#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SSEEnd {
    pub tokens_out: i32,
    pub decode_ms: i64,
}

#[cfg_attr(feature = "wasm", derive(Tsify))]
#[cfg_attr(feature = "wasm", tsify(into_wasm_abi, from_wasm_abi))]
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SSEError {
    pub code: ErrorKind,
    pub message: String,
    pub engine: Option<Engine>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SSEEvent {
    Started(SSEStarted),
    Token(SSEToken),
    Metrics(SSEMetrics),
    End(SSEEnd),
    Error(SSEError),
}
