//! Platform-agnostic inference backend trait
//!
//! This trait abstracts the inference execution layer, allowing worker-http
//! to be platform-independent. Different workers (CUDA, Metal, etc.) implement
//! this trait to provide their specific inference capabilities.
//!
//! Modified by: TEAM-017 (changed to &mut self for stateful models)

use crate::common::{InferenceResult, SamplingConfig};
use async_trait::async_trait;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Platform-agnostic inference backend
///
/// Implementations provide the actual inference execution (CUDA, Metal, etc.)
/// while worker-http handles HTTP/SSE concerns.
///
/// TEAM-017: Changed to &mut self to support stateful models with KV caches
#[async_trait]
pub trait InferenceBackend: Send + Sync {
    /// Execute inference with the given prompt and configuration
    ///
    /// Returns the complete inference result including tokens and stop reason.
    /// TEAM-017: Changed to &mut self for stateful model backends
    async fn execute(
        &mut self,
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
///
/// TEAM-017: Wrapped backend in Mutex to support &mut self in execute
pub struct AppState<B: InferenceBackend> {
    pub backend: Arc<Mutex<B>>,
}

impl<B: InferenceBackend> AppState<B> {
    pub fn new(backend: Arc<Mutex<B>>) -> Self {
        Self { backend }
    }
}
