//! Job routing and operation dispatch
//!
//! TEAM-186: Centralized job routing logic for all operations
//!
//! This module handles:
//! - Parsing operation payloads into typed Operation enum
//! - Routing operations to appropriate handlers
//! - Job lifecycle management (create, register, execute)
//!
//! # Architecture
//!
//! ```text
//! POST /v1/jobs (JSON payload)
//!     ↓
//! job_router::route_job()
//!     ↓
//! Parse into Operation enum
//!     ↓
//! Match and dispatch to handler
//!     ↓
//! Execute async in background
//!     ↓
//! Stream results via SSE
//! ```

use anyhow::Result;
use job_registry::JobRegistry;
use observability_narration_core::Narration;
use queen_rbee_hive_catalog::HiveCatalog;
use rbee_operations::Operation;
use std::sync::Arc;

const ACTOR_QUEEN_ROUTER: &str = "👑 queen-router";
const ACTION_ROUTE_JOB: &str = "route_job";
const ACTION_PARSE_OPERATION: &str = "parse_operation";

/// State required for job routing and execution
#[derive(Clone)]
pub struct JobState {
    pub registry: Arc<JobRegistry<String>>,
    pub hive_catalog: Arc<HiveCatalog>,
}

/// Response from job creation
#[derive(Debug, serde::Serialize)]
pub struct JobResponse {
    pub job_id: String,
    pub sse_url: String,
}

/// Create a new job and store its payload
///
/// This is the clean public API for job creation.
/// Called by HTTP layer to create jobs.
pub async fn create_job(
    state: JobState,
    payload: serde_json::Value,
) -> Result<JobResponse> {
    let job_id = state.registry.create_job();
    let sse_url = format!("/v1/jobs/{}/stream", job_id);
    
    state.registry.set_payload(&job_id, payload);
    
    Narration::new(ACTOR_QUEEN_ROUTER, "job_create", &job_id)
        .human(format!("Job {} created, waiting for client connection", job_id))
        .emit();
    
    Ok(JobResponse { job_id, sse_url })
}

/// Execute a job by retrieving its payload and streaming results
///
/// This is the clean public API for job execution.
/// Called by HTTP layer when client connects to SSE stream.
pub async fn execute_job(
    job_id: String,
    state: JobState,
) -> impl futures::stream::Stream<Item = String> {
    let registry = state.registry.clone();
    let hive_catalog = state.hive_catalog.clone();
    
    job_registry::execute_and_stream(
        job_id,
        registry.clone(),
        move |_job_id, payload| {
            route_operation(payload, registry, hive_catalog)
        },
    ).await
}

/// Internal: Route operation to appropriate handler
///
/// This parses the payload and dispatches to the correct operation handler.
async fn route_operation(
    payload: serde_json::Value,
    registry: Arc<JobRegistry<String>>,
    hive_catalog: Arc<HiveCatalog>,
) -> Result<()> {
    let state = JobState { registry, hive_catalog };
    // Parse payload into typed Operation enum
    let operation: Operation = serde_json::from_value(payload)
        .map_err(|e| anyhow::anyhow!("Failed to parse operation: {}", e))?;

    let operation_name = operation.name();

    Narration::new(ACTOR_QUEEN_ROUTER, ACTION_ROUTE_JOB, operation_name)
        .human(format!("Executing operation: {}", operation_name))
        .emit();

    // TEAM-186: Route to appropriate handler based on operation type
    match operation {
        // Hive operations
        Operation::HiveStart { hive_id } => {
            handle_hive_start_job(state, hive_id).await?;
        }
        Operation::HiveStop { hive_id } => {
            handle_hive_stop_job(state, hive_id).await?;
        }
        Operation::HiveList => {
            handle_hive_list_job(state).await?;
        }
        Operation::HiveGet { id } => {
            handle_hive_get_job(state, id).await?;
        }
        Operation::HiveCreate { host, port } => {
            handle_hive_create_job(state, host, port).await?;
        }
        Operation::HiveUpdate { id } => {
            handle_hive_update_job(state, id).await?;
        }
        Operation::HiveDelete { id } => {
            handle_hive_delete_job(state, id).await?;
        }

        // Worker operations
        Operation::WorkerSpawn { hive_id, model, worker, device } => {
            handle_worker_spawn_job(state, hive_id, model, worker, device).await?;
        }
        Operation::WorkerList { hive_id } => {
            handle_worker_list_job(state, hive_id).await?;
        }
        Operation::WorkerGet { hive_id, id } => {
            handle_worker_get_job(state, hive_id, id).await?;
        }
        Operation::WorkerDelete { hive_id, id } => {
            handle_worker_delete_job(state, hive_id, id).await?;
        }

        // Model operations
        Operation::ModelDownload { hive_id, model } => {
            handle_model_download_job(state, hive_id, model).await?;
        }
        Operation::ModelList { hive_id } => {
            handle_model_list_job(state, hive_id).await?;
        }
        Operation::ModelGet { hive_id, id } => {
            handle_model_get_job(state, hive_id, id).await?;
        }
        Operation::ModelDelete { hive_id, id } => {
            handle_model_delete_job(state, hive_id, id).await?;
        }

        // Inference operation
        Operation::Infer {
            hive_id,
            model,
            prompt,
            max_tokens,
            temperature,
            top_p,
            top_k,
            device,
            worker_id,
            stream,
        } => {
            handle_infer_job(
                state,
                hive_id,
                model,
                prompt,
                max_tokens,
                temperature,
                top_p,
                top_k,
                device,
                worker_id,
                stream,
            )
            .await?;
        }
    }

    Ok(())
}

// ============================================================================
// HIVE OPERATION HANDLERS
// ============================================================================

async fn handle_hive_start_job(
    _state: JobState,
    hive_id: String,
) -> Result<()> {
    // TODO: Implement hive start logic
    Narration::new(ACTOR_QUEEN_ROUTER, "hive_start", &hive_id)
        .human(format!("TODO: Start hive {}", hive_id))
        .emit();
    Ok(())
}

async fn handle_hive_stop_job(
    _state: JobState,
    hive_id: String,
) -> Result<()> {
    // TODO: Implement hive stop logic
    Narration::new(ACTOR_QUEEN_ROUTER, "hive_stop", &hive_id)
        .human(format!("TODO: Stop hive {}", hive_id))
        .emit();
    Ok(())
}

async fn handle_hive_list_job(_state: JobState) -> Result<()> {
    // TODO: Implement hive list logic
    Narration::new(ACTOR_QUEEN_ROUTER, "hive_list", "list")
        .human("TODO: List hives")
        .emit();
    Ok(())
}

async fn handle_hive_get_job(_state: JobState, id: String) -> Result<()> {
    // TODO: Implement hive get logic
    Narration::new(ACTOR_QUEEN_ROUTER, "hive_get", &id)
        .human(format!("TODO: Get hive {}", id))
        .emit();
    Ok(())
}

async fn handle_hive_create_job(
    _state: JobState,
    host: String,
    port: u16,
) -> Result<()> {
    // TODO: Implement hive create logic
    Narration::new(ACTOR_QUEEN_ROUTER, "hive_create", &host)
        .human(format!("TODO: Create hive at {}:{}", host, port))
        .emit();
    Ok(())
}

async fn handle_hive_update_job(_state: JobState, id: String) -> Result<()> {
    // TODO: Implement hive update logic
    Narration::new(ACTOR_QUEEN_ROUTER, "hive_update", &id)
        .human(format!("TODO: Update hive {}", id))
        .emit();
    Ok(())
}

async fn handle_hive_delete_job(_state: JobState, id: String) -> Result<()> {
    // TODO: Implement hive delete logic
    Narration::new(ACTOR_QUEEN_ROUTER, "hive_delete", &id)
        .human(format!("TODO: Delete hive {}", id))
        .emit();
    Ok(())
}

// ============================================================================
// WORKER OPERATION HANDLERS
// ============================================================================

async fn handle_worker_spawn_job(
    _state: JobState,
    hive_id: String,
    model: String,
    worker: String,
    device: u32,
) -> Result<()> {
    // TODO: Implement worker spawn logic
    Narration::new(ACTOR_QUEEN_ROUTER, "worker_spawn", &hive_id)
        .human(format!(
            "TODO: Spawn worker on hive {} with model {} ({}:device{})",
            hive_id, model, worker, device
        ))
        .emit();
    Ok(())
}

async fn handle_worker_list_job(
    _state: JobState,
    hive_id: String,
) -> Result<()> {
    // TODO: Implement worker list logic
    Narration::new(ACTOR_QUEEN_ROUTER, "worker_list", &hive_id)
        .human(format!("TODO: List workers on hive {}", hive_id))
        .emit();
    Ok(())
}

async fn handle_worker_get_job(
    _state: JobState,
    hive_id: String,
    id: String,
) -> Result<()> {
    // TODO: Implement worker get logic
    Narration::new(ACTOR_QUEEN_ROUTER, "worker_get", &id)
        .human(format!(
            "TODO: Get worker {} on hive {}",
            id, hive_id
        ))
        .emit();
    Ok(())
}

async fn handle_worker_delete_job(
    _state: JobState,
    hive_id: String,
    id: String,
) -> Result<()> {
    // TODO: Implement worker delete logic
    Narration::new(ACTOR_QUEEN_ROUTER, "worker_delete", &id)
        .human(format!(
            "TODO: Delete worker {} on hive {}",
            id, hive_id
        ))
        .emit();
    Ok(())
}

// ============================================================================
// MODEL OPERATION HANDLERS
// ============================================================================

async fn handle_model_download_job(
    _state: JobState,
    hive_id: String,
    model: String,
) -> Result<()> {
    // TODO: Implement model download logic
    Narration::new(ACTOR_QUEEN_ROUTER, "model_download", &model)
        .human(format!(
            "TODO: Download model {} on hive {}",
            model, hive_id
        ))
        .emit();
    Ok(())
}

async fn handle_model_list_job(
    _state: JobState,
    hive_id: String,
) -> Result<()> {
    // TODO: Implement model list logic
    Narration::new(ACTOR_QUEEN_ROUTER, "model_list", &hive_id)
        .human(format!("TODO: List models on hive {}", hive_id))
        .emit();
    Ok(())
}

async fn handle_model_get_job(
    _state: JobState,
    hive_id: String,
    id: String,
) -> Result<()> {
    // TODO: Implement model get logic
    Narration::new(ACTOR_QUEEN_ROUTER, "model_get", &id)
        .human(format!(
            "TODO: Get model {} on hive {}",
            id, hive_id
        ))
        .emit();
    Ok(())
}

async fn handle_model_delete_job(
    _state: JobState,
    hive_id: String,
    id: String,
) -> Result<()> {
    // TODO: Implement model delete logic
    Narration::new(ACTOR_QUEEN_ROUTER, "model_delete", &id)
        .human(format!(
            "TODO: Delete model {} on hive {}",
            id, hive_id
        ))
        .emit();
    Ok(())
}

// ============================================================================
// INFERENCE OPERATION HANDLER
// ============================================================================

#[allow(clippy::too_many_arguments)]
async fn handle_infer_job(
    _state: JobState,
    hive_id: String,
    model: String,
    prompt: String,
    max_tokens: u32,
    temperature: f32,
    _top_p: Option<f32>,
    _top_k: Option<u32>,
    _device: Option<String>,
    _worker_id: Option<String>,
    _stream: bool,
) -> Result<()> {
    // TODO: Implement inference logic
    Narration::new(ACTOR_QUEEN_ROUTER, "infer", &hive_id)
        .human(format!(
            "TODO: Run inference on hive {} with model {} (prompt: '{}', max_tokens: {}, temp: {})",
            hive_id, model, prompt, max_tokens, temperature
        ))
        .emit();
    Ok(())
}
