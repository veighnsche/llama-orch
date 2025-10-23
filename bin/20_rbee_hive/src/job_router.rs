//! Job routing and operation dispatch for rbee-hive
//!
//! TEAM-261: Mirrors queen-rbee pattern for consistency
//!
//! This module handles:
//! - Parsing operation payloads into typed Operation enum
//! - Routing operations to appropriate handlers
//! - Job lifecycle management (create, register, execute)
//!
//! # Architecture
//!
//! ```text
//! POST /v1/jobs (JSON payload from queen-rbee)
//!     ↓
//! job_router::route_job()
//!     ↓
//! Parse into Operation enum
//!     ↓
//! Match and dispatch to handler
//!     ↓
//! Execute async in background
//!     ↓
//! Stream results via SSE back to queen
//! ```

use anyhow::Result;
use job_server::JobRegistry;
use observability_narration_core::NarrationFactory;
use rbee_operations::Operation;
use std::sync::Arc;

// TEAM-261: Narration factory for hive job router
const NARRATE: NarrationFactory = NarrationFactory::new("hv-router");

/// State required for job routing and execution
#[derive(Clone)]
pub struct JobState {
    pub registry: Arc<JobRegistry<String>>,
    // TODO: Add worker_registry when implemented
    // TODO: Add model_catalog when implemented
}

/// Response from job creation
#[derive(Debug, serde::Serialize)]
pub struct JobResponse {
    pub job_id: String,
    pub sse_url: String,
}

/// Create a new job and store its payload
///
/// TEAM-261: Mirrors queen-rbee pattern
/// Called by HTTP layer to create jobs.
pub async fn create_job(state: JobState, payload: serde_json::Value) -> Result<JobResponse> {
    let job_id = state.registry.create_job();
    let sse_url = format!("/v1/jobs/{}/stream", job_id);

    state.registry.set_payload(&job_id, payload);

    // Create job-specific SSE channel for isolation
    observability_narration_core::sse_sink::create_job_channel(job_id.clone(), 1000);

    NARRATE
        .action("job_create")
        .context(&job_id)
        .job_id(&job_id)
        .human("Job {} created, waiting for client connection")
        .emit();

    Ok(JobResponse { job_id, sse_url })
}

/// Execute a job by retrieving its payload and streaming results
///
/// TEAM-261: Mirrors queen-rbee pattern
/// Called by HTTP layer when client connects to SSE stream.
pub async fn execute_job(
    job_id: String,
    state: JobState,
) -> impl futures::stream::Stream<Item = String> {
    let registry = state.registry.clone();

    job_server::execute_and_stream(job_id, registry.clone(), move |job_id, payload| {
        route_operation(job_id, payload, registry)
    })
    .await
}

/// Internal: Route operation to appropriate handler
///
/// TEAM-261: Parse payload and dispatch to worker/model handlers
async fn route_operation(
    job_id: String,
    payload: serde_json::Value,
    _registry: Arc<JobRegistry<String>>,
) -> Result<()> {
    // Parse payload into typed Operation enum
    let operation: Operation = serde_json::from_value(payload)
        .map_err(|e| anyhow::anyhow!("Failed to parse operation: {}", e))?;

    let operation_name = operation.name();

    NARRATE
        .action("route_job")
        .context(operation_name)
        .job_id(&job_id)
        .human("Executing operation: {}")
        .emit();

    // ============================================================================
    // OPERATION ROUTING
    // ============================================================================
    //
    // Hive handles Worker and Model operations
    // (Hive operations like HiveStart/HiveStop are handled by queen-rbee)
    //
    match operation {
        // Worker operations
        Operation::WorkerSpawn { hive_id, model, worker, device } => {
            NARRATE
                .action("worker_spawn")
                .job_id(&job_id)
                .context(&hive_id)
                .context(&model)
                .context(&worker)
                .context(device.to_string())
                .human("TODO: Spawn worker on hive '{}' with model '{}', worker '{}', device {}")
                .emit();

            // TODO: Implement worker spawning
            // - Validate model exists
            // - Validate device exists
            // - Spawn worker process
            // - Register in worker_registry
        }

        Operation::WorkerList { hive_id } => {
            NARRATE
                .action("worker_list")
                .job_id(&job_id)
                .context(&hive_id)
                .human("TODO: List workers on hive '{}'")
                .emit();

            // TODO: Implement worker listing
            // - Query worker_registry
            // - Format as JSON table
        }

        Operation::WorkerGet { hive_id, id } => {
            NARRATE
                .action("worker_get")
                .job_id(&job_id)
                .context(&hive_id)
                .context(&id)
                .human("TODO: Get worker '{}' on hive '{}'")
                .emit();

            // TODO: Implement worker get
            // - Query worker_registry
            // - Return worker details
        }

        Operation::WorkerDelete { hive_id, id } => {
            NARRATE
                .action("worker_delete")
                .job_id(&job_id)
                .context(&hive_id)
                .context(&id)
                .human("TODO: Delete worker '{}' on hive '{}'")
                .emit();

            // TODO: Implement worker deletion
            // - Stop worker process
            // - Remove from worker_registry
        }

        // Model operations
        Operation::ModelDownload { hive_id, model } => {
            NARRATE
                .action("model_download")
                .job_id(&job_id)
                .context(&hive_id)
                .context(&model)
                .human("TODO: Download model '{}' on hive '{}'")
                .emit();

            // TODO: Implement model download
            // - Download model files
            // - Register in model_catalog
        }

        Operation::ModelList { hive_id } => {
            NARRATE
                .action("model_list")
                .job_id(&job_id)
                .context(&hive_id)
                .human("TODO: List models on hive '{}'")
                .emit();

            // TODO: Implement model listing
            // - Query model_catalog
            // - Format as JSON table
        }

        Operation::ModelGet { hive_id, id } => {
            NARRATE
                .action("model_get")
                .job_id(&job_id)
                .context(&hive_id)
                .context(&id)
                .human("TODO: Get model '{}' on hive '{}'")
                .emit();

            // TODO: Implement model get
            // - Query model_catalog
            // - Return model details
        }

        Operation::ModelDelete { hive_id, id } => {
            NARRATE
                .action("model_delete")
                .job_id(&job_id)
                .context(&hive_id)
                .context(&id)
                .human("TODO: Delete model '{}' on hive '{}'")
                .emit();

            // TODO: Implement model deletion
            // - Delete model files
            // - Remove from model_catalog
        }

        // ========================================================================
        // INFERENCE REJECTION - CRITICAL ARCHITECTURE NOTE (TEAM-261)
        // ========================================================================
        //
        // ⚠️  INFER SHOULD NOT BE IN HIVE!
        //
        // Why?
        // - Hive only manages worker LIFECYCLE (spawn/stop/list)
        // - Queen handles inference routing DIRECTLY to workers
        // - Queen → Worker is DIRECT HTTP (circumvents hive)
        // - This is INTENTIONAL for performance and simplicity
        //
        // If you see Infer here, something is wrong with the routing in queen-rbee!
        //
        // See: bin/.plan/TEAM_261_ARCHITECTURE_CLARITY.md
        //
        Operation::Infer { .. } => {
            return Err(anyhow::anyhow!(
                "Infer operation should NOT be routed to hive! \
                 Queen should route inference directly to workers. \
                 This indicates a routing bug in queen-rbee/src/job_router.rs. \
                 See bin/.plan/TEAM_261_ARCHITECTURE_CLARITY.md for details."
            ));
        }

        // Unsupported operations (handled by queen-rbee)
        _ => {
            return Err(anyhow::anyhow!(
                "Operation '{}' is not supported by rbee-hive (should be handled by queen-rbee)",
                operation_name
            ));
        }
    }

    Ok(())
}
