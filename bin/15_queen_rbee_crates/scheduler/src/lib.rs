// TEAM-275: Scheduler crate (stub for future Rhai scheduler)
#![warn(missing_docs)]
#![warn(clippy::all)]

//! queen-rbee-scheduler
//!
//! **Category:** Orchestration
//! **Pattern:** Strategy Pattern (pluggable scheduler implementations)
//! **Future:** M2 - Rhai programmable scheduler
//!
//! # Purpose
//!
//! General-purpose job scheduler for rbee. Routes jobs to appropriate workers based on:
//! - **Job type:** LLM inference, image generation, batch processing, distributed inference
//! - **Worker type:** Bespoke workers (Candle), adapters (llama.cpp, vLLM, ComfyUI)
//! - **Resource requirements:** VRAM, CPU, multi-GPU
//! - **Custom policies:** Cost, latency, geo-location, compliance
//!
//! # Current Implementation (M0/M1)
//!
//! Simple first-available scheduler:
//! - Find workers serving requested model
//! - Filter by online + available status
//! - Return first match (no load balancing)
//!
//! # Future Implementation (M2+)
//!
//! Rhai programmable scheduler:
//! - User-written Rhai scripts for custom routing
//! - 40+ built-in helper functions
//! - YAML config support (compiles to Rhai)
//! - Web UI policy builder
//! - Platform mode (immutable) vs Home/Lab mode (customizable)
//! - Multi-modal routing (LLM, image-gen, batch, distributed)
//!
//! See: `.business/stakeholders/RHAI_PROGRAMMABLE_SCHEDULER.md`
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │  JobScheduler (trait)                   │
//! └─────────────────┬───────────────────────┘
//!                   │
//!        ┌──────────┴──────────┐
//!        │                     │
//! ┌──────▼──────┐      ┌──────▼──────────┐
//! │ SimpleScheduler │  │ RhaiScheduler   │
//! │ (M0/M1)        │  │ (M2+)           │
//! └────────────────┘  └─────────────────┘
//! ```

mod simple;
mod types;

pub use simple::SimpleScheduler;
pub use types::{JobRequest, ScheduleResult, SchedulerError};

// TODO M2: Add Rhai scheduler
// pub mod rhai;
// pub use rhai::RhaiScheduler;

/// Job scheduler trait
///
/// Implementations decide which worker should handle a job request.
///
/// Supports multiple job types:
/// - LLM inference (text generation)
/// - Image generation (Stable Diffusion, ComfyUI)
/// - Batch processing (vLLM)
/// - Distributed inference (multi-GPU)
///
/// # Implementations
///
/// - `SimpleScheduler` - First available worker (M0/M1)
/// - `RhaiScheduler` - Programmable Rhai scripts (M2+)
///
/// # Example
///
/// ```rust,no_run
/// use queen_rbee_scheduler::{SimpleScheduler, JobRequest};
/// use queen_rbee_telemetry_registry::TelemetryRegistry; // TEAM-374
/// use std::sync::Arc;
///
/// # async fn example() -> anyhow::Result<()> {
/// let registry = Arc::new(TelemetryRegistry::new()); // TEAM-374
/// let scheduler = SimpleScheduler::new(registry);
///
/// let request = JobRequest {
///     job_id: "job-123".to_string(),
///     model: "meta-llama/Llama-3-8b".to_string(),
///     prompt: "Hello!".to_string(),
///     max_tokens: 20,
///     temperature: 0.7,
///     top_p: None,
///     top_k: None,
/// };
///
/// let result = scheduler.schedule(request).await?;
/// println!("Selected worker: {}", result.worker_id);
/// # Ok(())
/// # }
/// ```
#[async_trait::async_trait]
pub trait JobScheduler: Send + Sync {
    /// Schedule a job request
    ///
    /// Finds the best worker for the request and returns routing information.
    ///
    /// # Arguments
    ///
    /// * `request` - Job request with model, prompt, parameters
    ///
    /// # Returns
    ///
    /// * `Ok(ScheduleResult)` - Worker selected, contains worker_id and URL
    /// * `Err(SchedulerError)` - No workers available or scheduling failed
    async fn schedule(&self, request: JobRequest) -> Result<ScheduleResult, SchedulerError>;
}
