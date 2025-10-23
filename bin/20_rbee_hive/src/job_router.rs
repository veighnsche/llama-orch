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
use rbee_hive_model_catalog::{ModelCatalog, ModelProvisioner}; // TEAM-268: Model catalog, TEAM-269: Model provisioner
use rbee_operations::Operation;
use std::sync::Arc;

// TEAM-261: Narration factory for hive job router
const NARRATE: NarrationFactory = NarrationFactory::new("hv-router");

/// State required for job routing and execution
#[derive(Clone)]
pub struct JobState {
    pub registry: Arc<JobRegistry<String>>,
    pub model_catalog: Arc<ModelCatalog>, // TEAM-268: Model catalog
    pub model_provisioner: Arc<ModelProvisioner>, // TEAM-269: Model provisioner
    // TODO: Add worker_registry when implemented
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
    let state_clone = state.clone(); // TEAM-268: Clone full state for closure

    job_server::execute_and_stream(job_id, registry.clone(), move |job_id, payload| {
        route_operation(job_id, payload, state_clone.clone())
    })
    .await
}

/// Internal: Route operation to appropriate handler
///
/// TEAM-261: Parse payload and dispatch to worker/model handlers
async fn route_operation(
    job_id: String,
    payload: serde_json::Value,
    state: JobState, // TEAM-268: Changed from _registry to full state
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
            // TEAM-269: Implemented model download with provisioner
            NARRATE
                .action("model_download_start")
                .job_id(&job_id)
                .context(&hive_id)
                .context(&model)
                .human("📥 Downloading model '{}' on hive '{}'")
                .emit();

            // Check if model already exists
            if state.model_catalog.contains(&model) {
                NARRATE
                    .action("model_download_exists")
                    .job_id(&job_id)
                    .context(&model)
                    .human("⚠️  Model '{}' already exists in catalog")
                    .emit();
                
                return Err(anyhow::anyhow!("Model '{}' already exists", model));
            }

            // Check if vendor supports this model
            if !state.model_provisioner.is_supported(&model) {
                NARRATE
                    .action("model_download_unsupported")
                    .job_id(&job_id)
                    .context(&model)
                    .human("❌ No vendor supports model '{}'. Supported formats: HuggingFace (contains '/')")
                    .emit();
                
                return Err(anyhow::anyhow!(
                    "No vendor supports model '{}'. Supported formats: HuggingFace (contains '/')",
                    model
                ));
            }

            // Download model using provisioner
            match state.model_provisioner.download_model(&job_id, &model).await {
                Ok(model_id) => {
                    NARRATE
                        .action("model_download_complete")
                        .job_id(&job_id)
                        .context(&model_id)
                        .human("✅ Model '{}' downloaded successfully")
                        .emit();
                }
                Err(e) => {
                    NARRATE
                        .action("model_download_failed")
                        .job_id(&job_id)
                        .context(&model)
                        .context(&e.to_string())
                        .human("❌ Failed to download model '{}': {}")
                        .emit();
                    
                    return Err(e);
                }
            }
        }

        Operation::ModelList { hive_id } => {
            // TEAM-268: Implemented model list
            NARRATE
                .action("model_list_start")
                .job_id(&job_id)
                .context(&hive_id)
                .human("📋 Listing models on hive '{}'")
                .emit();

            let models = state.model_catalog.list();

            NARRATE
                .action("model_list_result")
                .job_id(&job_id)
                .context(&models.len().to_string())
                .human("Found {} model(s)")
                .emit();

            // Format as JSON table
            if models.is_empty() {
                NARRATE
                    .action("model_list_empty")
                    .job_id(&job_id)
                    .human("No models found")
                    .emit();
            } else {
                for model in &models {
                    let size_gb = model.size_bytes as f64 / 1_000_000_000.0;
                    let status = match &model.status {
                        rbee_hive_model_catalog::ModelStatus::Ready => "ready",
                        rbee_hive_model_catalog::ModelStatus::Downloading { progress } => {
                            &format!("downloading ({:.0}%)", progress * 100.0)
                        }
                        rbee_hive_model_catalog::ModelStatus::Failed { .. } => "failed",
                    };

                    NARRATE
                        .action("model_list_entry")
                        .job_id(&job_id)
                        .context(&model.id)
                        .context(&model.name)
                        .context(&format!("{:.2} GB", size_gb))
                        .context(status)
                        .human("  {} | {} | {} | {}")
                        .emit();
                }
            }
        }

        Operation::ModelGet { hive_id, id } => {
            // TEAM-268: Implemented model get
            NARRATE
                .action("model_get_start")
                .job_id(&job_id)
                .context(&hive_id)
                .context(&id)
                .human("🔍 Getting model '{}' on hive '{}'")
                .emit();

            match state.model_catalog.get(&id) {
                Ok(model) => {
                    NARRATE
                        .action("model_get_found")
                        .job_id(&job_id)
                        .context(&model.id)
                        .context(&model.name)
                        .context(&model.path.display().to_string())
                        .human("✅ Model: {} | Name: {} | Path: {}")
                        .emit();

                    // Emit model details as JSON
                    let json = serde_json::to_string_pretty(&model)
                        .unwrap_or_else(|_| "Failed to serialize".to_string());

                    NARRATE
                        .action("model_get_details")
                        .job_id(&job_id)
                        .human(&json)
                        .emit();
                }
                Err(e) => {
                    NARRATE
                        .action("model_get_error")
                        .job_id(&job_id)
                        .context(&id)
                        .context(&e.to_string())
                        .human("❌ Model '{}' not found: {}")
                        .emit();
                    return Err(e);
                }
            }
        }

        Operation::ModelDelete { hive_id, id } => {
            // TEAM-268: Implemented model delete
            NARRATE
                .action("model_delete_start")
                .job_id(&job_id)
                .context(&hive_id)
                .context(&id)
                .human("🗑️  Deleting model '{}' on hive '{}'")
                .emit();

            match state.model_catalog.remove(&id) {
                Ok(model) => {
                    NARRATE
                        .action("model_delete_catalog")
                        .job_id(&job_id)
                        .context(&id)
                        .human("✅ Removed '{}' from catalog")
                        .emit();

                    // NOTE: Catalog remove() already deletes the directory
                    NARRATE
                        .action("model_delete_files")
                        .job_id(&job_id)
                        .context(&model.path.display().to_string())
                        .human("✅ Deleted model directory: {}")
                        .emit();
                }
                Err(e) => {
                    NARRATE
                        .action("model_delete_error")
                        .job_id(&job_id)
                        .context(&id)
                        .context(&e.to_string())
                        .human("❌ Failed to delete model '{}': {}")
                        .emit();
                    return Err(e);
                }
            }
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
