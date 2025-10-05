//! Platform-agnostic inference backend trait
//!
//! This trait abstracts the inference execution layer, allowing worker-http
//! to be platform-independent. Different workers (CUDA, Metal, etc.) implement
//! this trait to provide their specific inference capabilities.

use worker_common::{InferenceResult, SamplingConfig};
use async_trait::async_trait;
use std::sync::Arc;

/// Platform-agnostic inference backend
///
/// Implementations provide the actual inference execution (CUDA, Metal, etc.)
/// while worker-http handles HTTP/SSE concerns.
#[async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Execute inference with the given prompt and configuration
    ///
    /// Returns the complete inference result including tokens and stop reason.
    async fn execute(
        &self,
        prompt: &str,
        config: &SamplingConfig,
    ) -> Result<InferenceResult, Box<dyn std::error::Error + Send + Sync>>;
    
    /// Cancel an in-flight inference by job ID
    ///
    /// This is a best-effort operation - the inference may complete before cancellation.
    async fn cancel(&self, job_id: &str) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
    
    /// Get current VRAM usage in bytes
    fn vram_usage(&self) -> u64;
    
    /// Check if backend is healthy and ready for inference
    fn is_healthy(&self) -> bool;
}

/// Shared application state for HTTP handlers
pub struct AppState<B: InferenceBackend> {
    pub backend: Arc<B>,
}

impl<B: InferenceBackend> AppState<B> {
    pub fn new(backend: Arc<B>) -> Self {
        Self { backend }
    }
}
