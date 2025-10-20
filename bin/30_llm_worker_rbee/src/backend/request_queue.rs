// TEAM-149: Created for real-time streaming implementation
//! Request queue for decoupling HTTP handlers from generation
//!
//! This module implements the request queue pattern from candle-vllm:
//! - HTTP handlers add requests to queue and return immediately
//! - Generation engine processes requests sequentially
//! - Tokens flow through channels to SSE streams
//!
//! Reference: reference/candle-vllm/src/openai/openai_server.rs

use crate::common::SamplingConfig;
use tokio::sync::mpsc;

/// A generation request from an HTTP handler
#[derive(Debug)]
pub struct GenerationRequest {
    /// Unique request ID (job_id from HTTP request)
    pub request_id: String,

    /// Input prompt to generate from
    pub prompt: String,

    /// Sampling configuration (temperature, top_p, etc.)
    pub config: SamplingConfig,

    /// Channel to send token responses back to HTTP handler
    pub response_tx: mpsc::UnboundedSender<TokenResponse>,
}

/// Response sent from generation engine to HTTP handler
#[derive(Debug, Clone)]
pub enum TokenResponse {
    /// A generated token (decoded text)
    Token(String),

    /// An error occurred during generation
    Error(String),

    /// Generation completed successfully
    Done,
}

/// Request queue for adding generation requests
///
/// This is passed to HTTP handlers instead of the backend directly.
/// Handlers add requests to the queue and return immediately with a stream.
#[derive(Clone)]
pub struct RequestQueue {
    tx: mpsc::UnboundedSender<GenerationRequest>,
}

impl RequestQueue {
    /// Create a new request queue
    ///
    /// Returns the queue (for HTTP handlers) and receiver (for generation engine)
    pub fn new() -> (Self, mpsc::UnboundedReceiver<GenerationRequest>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self { tx }, rx)
    }

    /// Add a request to the queue
    ///
    /// Returns Ok(()) if request was queued successfully.
    /// Returns Err if the generation engine has stopped.
    pub fn add_request(&self, request: GenerationRequest) -> Result<(), String> {
        self.tx
            .send(request)
            .map_err(|e| format!("Queue send failed (generation engine stopped): {}", e))
    }
}

impl Default for RequestQueue {
    fn default() -> Self {
        Self::new().0
    }
}
