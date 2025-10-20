//! Scheduler for queen-rbee
//!
//! **Category:** Orchestration
//! **Pattern:** Command Pattern
//! **Standard:** See `/bin/CRATE_INTERFACE_STANDARD.md`
//!
//! TEAM-164: Migrated job orchestration logic from http.rs
//!
//! This crate provides device selection and scheduling logic for distributing
//! inference tasks across available hives and devices.
//!
//! # Interface
//!
//! ## Request Type
//! ```rust
//! pub struct JobRequest {
//!     pub model: String,
//!     pub prompt: String,
//!     pub max_tokens: u32,
//!     pub temperature: f32,
//! }
//! ```
//!
//! ## Response Type
//! ```rust
//! pub struct JobResponse {
//!     pub job_id: String,
//!     pub sse_url: String,
//! }
//! ```
//!
//! ## Entrypoint
//! ```rust
//! pub async fn orchestrate_job(
//!     registry: Arc<JobRegistry<String>>,
//!     catalog: Arc<HiveCatalog>,
//!     request: JobRequest,
//! ) -> Result<JobResponse>
//! ```

use anyhow::Result;
use job_registry::JobRegistry;
use observability_narration_core::Narration;
use queen_rbee_hive_catalog::HiveCatalog;
use std::sync::Arc;

const ACTOR_SCHEDULER: &str = "👑 scheduler";
const ACTION_ORCHESTRATE: &str = "orchestrate_job";

// ============================================================================
// REQUEST / RESPONSE TYPES (Command Pattern)
// ============================================================================

/// Request to orchestrate a job
#[derive(Debug, Clone)]
pub struct JobRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
}

/// Response from job orchestration
#[derive(Debug, Clone)]
pub struct JobResponse {
    /// Unique job identifier
    pub job_id: String,
    /// URL for SSE streaming
    pub sse_url: String,
}

// ============================================================================
// ORCHESTRATION LOGIC
// ============================================================================

/// Orchestrate a new job
///
/// **Pattern:** Command Pattern (see CRATE_INTERFACE_STANDARD.md)
///
/// TEAM-164: ALL job orchestration logic lives here
/// - Checks hive availability
/// - Creates job in registry
/// - Sets up token streaming
/// - Returns job response
///
/// # Arguments
/// * `registry` - Job registry for state management
/// * `catalog` - Hive catalog for finding available hives
/// * `request` - Job orchestration request
///
/// # Returns
/// * `Ok(JobResponse)` - Job orchestrated successfully
/// * `Err` - Failed to orchestrate job
pub async fn orchestrate_job(
    registry: Arc<JobRegistry<String>>,
    catalog: Arc<HiveCatalog>,
    request: JobRequest,
) -> Result<JobResponse> {
    let job_id = registry.create_job();

    Narration::new(ACTOR_SCHEDULER, ACTION_ORCHESTRATE, &job_id)
        .human(format!("Orchestrating job {} for model {}", &job_id, &request.model))
        .emit();

    // Check if any hives available
    let hives = catalog.list_hives().await?;

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
    registry.set_token_receiver(&job_id, rx);

    if hives.is_empty() {
        Narration::new(ACTOR_SCHEDULER, ACTION_ORCHESTRATE, &job_id)
            .human("No hives found in catalog")
            .emit();
        
        let _ = tx.send("No hives found.".to_string());
        let _ = tx.send("[DONE]".to_string());
    } else {
        // TODO: Actual job orchestration logic
        // - Select best hive/device
        // - Forward job to hive
        // - Stream results back
        let _ = tx.send("Job created, orchestration not yet implemented".to_string());
        let _ = tx.send("[DONE]".to_string());
    }

    let sse_url = format!("/jobs/{}/stream", &job_id);
    
    // Return structured response (Command Pattern)
    Ok(JobResponse {
        job_id,
        sse_url,
    })
}
