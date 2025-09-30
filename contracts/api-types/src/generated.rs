// Deterministic template for generated API types matching contracts/openapi/data.yaml
use serde::{Deserialize, Serialize};
fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmissionStreams {
    pub sse: String,
    pub sse_verbose: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preparation {
    pub steps: Vec<PreparationStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreparationStep {
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Engine {
    Llamacpp,
    Vllm,
    Tgi,
    Triton,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Workload {
    Completion,
    Embedding,
    Rerank,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeterminismLevel {
    Strict,
    BestEffort,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Priority {
    Interactive,
    Batch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KVHint {
    Reuse,
    Cold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlacementMode {
    Pin,
    Prefer,
    Auto,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlacementOverrides {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<PlacementMode>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pin_pool_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefer_pools: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub avoid_pools: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_device_mask: Option<String>,
    #[serde(default = "default_true")]
    pub allow_fallback: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskRequest {
    pub task_id: String,
    pub session_id: String,
    pub workload: Workload,
    pub model_ref: String,
    pub engine: Engine,
    pub ctx: i32,
    pub priority: Priority,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub determinism: Option<DeterminismLevel>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampler_profile_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inputs: Option<Vec<String>>,
    pub max_tokens: i32,
    pub deadline_ms: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected_tokens: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_hint: Option<KVHint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement: Option<PlacementOverrides>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdmissionResponse {
    pub task_id: String,
    pub queue_position: i32,
    pub predicted_start_ms: i64,
    pub backoff_ms: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub streams: Option<AdmissionStreams>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preparation: Option<Preparation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorEnvelope {
    pub code: ErrorKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<Engine>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retriable: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_after_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub policy_label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttl_ms_remaining: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub turns: Option<i32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_bytes: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_warmth: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_budget_remaining: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_budget_remaining_ms: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cost_budget_remaining: Option<f64>,
}
