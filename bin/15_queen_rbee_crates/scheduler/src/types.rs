// TEAM-275: Scheduler types
use serde::{Deserialize, Serialize};

/// Job request for scheduling
///
/// General-purpose request that supports multiple job types:
/// - LLM inference (text generation)
/// - Image generation (Stable Diffusion, ComfyUI)
/// - Batch processing (vLLM)
/// - Distributed inference (multi-GPU)
#[derive(Debug, Clone)]
pub struct JobRequest {
    /// Job ID for tracking
    pub job_id: String,
    /// Model to use for inference
    pub model: String,
    /// Prompt text
    pub prompt: String,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-p for nucleus sampling
    pub top_p: Option<f32>,
    /// Top-k for top-k sampling
    pub top_k: Option<u32>,
}

/// Result of scheduling decision
#[derive(Debug, Clone)]
pub struct ScheduleResult {
    /// Selected worker ID
    pub worker_id: String,
    /// Worker base URL (e.g., "http://localhost:9001")
    pub worker_url: String,
    /// Worker port
    pub worker_port: u16,
    /// Model being served
    pub model: String,
    /// Device worker is using
    pub device: String,
}

/// Worker inference request payload (sent to worker)
#[derive(Debug, Serialize)]
pub struct WorkerInferenceRequest {
    /// Prompt text
    pub prompt: String,
    /// Maximum tokens to generate
    pub max_tokens: u32,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-p for nucleus sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Top-k for top-k sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
}

/// Worker job creation response
#[derive(Debug, Deserialize)]
pub struct WorkerJobResponse {
    /// Job ID created by worker
    pub job_id: String,
    /// SSE stream URL
    pub sse_url: String,
}

/// Scheduler error types
#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    /// No workers available for the requested model
    #[error("No available worker found for model '{model}'. Make sure a worker is running and has sent heartbeats to queen.")]
    NoWorkersAvailable {
        /// Model that was requested
        model: String,
    },

    /// Worker communication failed
    #[error("Failed to communicate with worker: {0}")]
    WorkerCommunicationFailed(String),

    /// Worker returned an error
    #[error("Worker returned error {status}: {message}")]
    WorkerError {
        /// HTTP status code
        status: u16,
        /// Error message from worker
        message: String,
    },

    /// Failed to parse worker response
    #[error("Failed to parse worker response: {0}")]
    ParseError(String),

    /// Stream connection failed
    #[error("Failed to connect to worker stream: {0}")]
    StreamConnectionFailed(String),

    /// Stream read error
    #[error("Error reading stream: {0}")]
    StreamReadError(String),

    /// Generic error
    #[error("{0}")]
    Other(String),
}

impl From<anyhow::Error> for SchedulerError {
    fn from(err: anyhow::Error) -> Self {
        SchedulerError::Other(err.to_string())
    }
}

impl From<reqwest::Error> for SchedulerError {
    fn from(err: reqwest::Error) -> Self {
        SchedulerError::WorkerCommunicationFailed(err.to_string())
    }
}
