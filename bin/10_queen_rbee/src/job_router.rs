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

/// State required for job routing
#[derive(Clone)]
pub struct JobRouterState {
    pub registry: Arc<JobRegistry<String>>,
    pub hive_catalog: Arc<HiveCatalog>,
}

/// Response from job creation
#[derive(Debug)]
pub struct JobResponse {
    pub job_id: String,
    pub sse_url: String,
}

/// Route a job based on its operation type
///
/// TEAM-186: Main entry point for all job operations
///
/// # Flow
/// 1. Parse JSON payload into Operation enum
/// 2. Create job in registry
/// 3. Route to appropriate handler based on operation type
/// 4. Execute handler async (streams results to job registry)
/// 5. Return job_id and sse_url to client
pub async fn route_job(
    state: JobRouterState,
    payload: serde_json::Value,
) -> Result<JobResponse> {
    // TEAM-186: Parse payload into typed Operation enum
    Narration::new(ACTOR_QUEEN_ROUTER, ACTION_PARSE_OPERATION, "parsing")
        .human("Parsing operation from payload")
        .emit();

    let operation: Operation = serde_json::from_value(payload)
        .map_err(|e| anyhow::anyhow!("Failed to parse operation: {}", e))?;

    let operation_name = operation.name();

    Narration::new(ACTOR_QUEEN_ROUTER, ACTION_ROUTE_JOB, operation_name)
        .human(format!("Routing job for operation: {}", operation_name))
        .emit();

    // TEAM-186: Create job in registry (generates job_id)
    let job_id = state.registry.create_job();
    let sse_url = format!("/v1/jobs/{}/stream", job_id);

    // TEAM-186: Create channel for streaming results
    let (_tx, rx) = tokio::sync::mpsc::unbounded_channel();
    state.registry.set_token_receiver(&job_id, rx);

    Narration::new(ACTOR_QUEEN_ROUTER, ACTION_ROUTE_JOB, &job_id)
        .human(format!("Job {} created for operation {}", job_id, operation_name))
        .emit();

    // TEAM-186: Route to appropriate handler based on operation type
    match operation {
        // Hive operations
        Operation::HiveStart { hive_id } => {
            handle_hive_start_job(state, job_id.clone(), hive_id).await?;
        }
        Operation::HiveStop { hive_id } => {
            handle_hive_stop_job(state, job_id.clone(), hive_id).await?;
        }
        Operation::HiveList => {
            handle_hive_list_job(state, job_id.clone()).await?;
        }
        Operation::HiveGet { id } => {
            handle_hive_get_job(state, job_id.clone(), id).await?;
        }
        Operation::HiveCreate { host, port } => {
            handle_hive_create_job(state, job_id.clone(), host, port).await?;
        }
        Operation::HiveUpdate { id } => {
            handle_hive_update_job(state, job_id.clone(), id).await?;
        }
        Operation::HiveDelete { id } => {
            handle_hive_delete_job(state, job_id.clone(), id).await?;
        }

        // Worker operations
        Operation::WorkerSpawn { hive_id, model, worker, device } => {
            handle_worker_spawn_job(state, job_id.clone(), hive_id, model, worker, device).await?;
        }
        Operation::WorkerList { hive_id } => {
            handle_worker_list_job(state, job_id.clone(), hive_id).await?;
        }
        Operation::WorkerGet { hive_id, id } => {
            handle_worker_get_job(state, job_id.clone(), hive_id, id).await?;
        }
        Operation::WorkerDelete { hive_id, id } => {
            handle_worker_delete_job(state, job_id.clone(), hive_id, id).await?;
        }

        // Model operations
        Operation::ModelDownload { hive_id, model } => {
            handle_model_download_job(state, job_id.clone(), hive_id, model).await?;
        }
        Operation::ModelList { hive_id } => {
            handle_model_list_job(state, job_id.clone(), hive_id).await?;
        }
        Operation::ModelGet { hive_id, id } => {
            handle_model_get_job(state, job_id.clone(), hive_id, id).await?;
        }
        Operation::ModelDelete { hive_id, id } => {
            handle_model_delete_job(state, job_id.clone(), hive_id, id).await?;
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
                job_id.clone(),
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

    Ok(JobResponse { job_id, sse_url })
}

// ============================================================================
// HIVE OPERATION HANDLERS
// ============================================================================

async fn handle_hive_start_job(
    _state: JobRouterState,
    job_id: String,
    hive_id: String,
) -> Result<()> {
    // TODO: Implement hive start logic
    Narration::new(ACTOR_QUEEN_ROUTER, "hive_start", &job_id)
        .human(format!("TODO: Start hive {} for job {}", hive_id, job_id))
        .emit();
    Ok(())
}

async fn handle_hive_stop_job(
    _state: JobRouterState,
    job_id: String,
    hive_id: String,
) -> Result<()> {
    // TODO: Implement hive stop logic
    Narration::new(ACTOR_QUEEN_ROUTER, "hive_stop", &job_id)
        .human(format!("TODO: Stop hive {} for job {}", hive_id, job_id))
        .emit();
    Ok(())
}

async fn handle_hive_list_job(_state: JobRouterState, job_id: String) -> Result<()> {
    // TODO: Implement hive list logic
    Narration::new(ACTOR_QUEEN_ROUTER, "hive_list", &job_id)
        .human(format!("TODO: List hives for job {}", job_id))
        .emit();
    Ok(())
}

async fn handle_hive_get_job(_state: JobRouterState, job_id: String, id: String) -> Result<()> {
    // TODO: Implement hive get logic
    Narration::new(ACTOR_QUEEN_ROUTER, "hive_get", &job_id)
        .human(format!("TODO: Get hive {} for job {}", id, job_id))
        .emit();
    Ok(())
}

async fn handle_hive_create_job(
    _state: JobRouterState,
    job_id: String,
    host: String,
    port: u16,
) -> Result<()> {
    // TODO: Implement hive create logic
    Narration::new(ACTOR_QUEEN_ROUTER, "hive_create", &job_id)
        .human(format!("TODO: Create hive at {}:{} for job {}", host, port, job_id))
        .emit();
    Ok(())
}

async fn handle_hive_update_job(_state: JobRouterState, job_id: String, id: String) -> Result<()> {
    // TODO: Implement hive update logic
    Narration::new(ACTOR_QUEEN_ROUTER, "hive_update", &job_id)
        .human(format!("TODO: Update hive {} for job {}", id, job_id))
        .emit();
    Ok(())
}

async fn handle_hive_delete_job(_state: JobRouterState, job_id: String, id: String) -> Result<()> {
    // TODO: Implement hive delete logic
    Narration::new(ACTOR_QUEEN_ROUTER, "hive_delete", &job_id)
        .human(format!("TODO: Delete hive {} for job {}", id, job_id))
        .emit();
    Ok(())
}

// ============================================================================
// WORKER OPERATION HANDLERS
// ============================================================================

#[allow(clippy::too_many_arguments)]
async fn handle_worker_spawn_job(
    _state: JobRouterState,
    job_id: String,
    hive_id: String,
    model: String,
    worker: String,
    device: u32,
) -> Result<()> {
    // TODO: Implement worker spawn logic
    Narration::new(ACTOR_QUEEN_ROUTER, "worker_spawn", &job_id)
        .human(format!(
            "TODO: Spawn worker on hive {} with model {} ({}:device{}) for job {}",
            hive_id, model, worker, device, job_id
        ))
        .emit();
    Ok(())
}

async fn handle_worker_list_job(
    _state: JobRouterState,
    job_id: String,
    hive_id: String,
) -> Result<()> {
    // TODO: Implement worker list logic
    Narration::new(ACTOR_QUEEN_ROUTER, "worker_list", &job_id)
        .human(format!("TODO: List workers on hive {} for job {}", hive_id, job_id))
        .emit();
    Ok(())
}

async fn handle_worker_get_job(
    _state: JobRouterState,
    job_id: String,
    hive_id: String,
    id: String,
) -> Result<()> {
    // TODO: Implement worker get logic
    Narration::new(ACTOR_QUEEN_ROUTER, "worker_get", &job_id)
        .human(format!(
            "TODO: Get worker {} on hive {} for job {}",
            id, hive_id, job_id
        ))
        .emit();
    Ok(())
}

async fn handle_worker_delete_job(
    _state: JobRouterState,
    job_id: String,
    hive_id: String,
    id: String,
) -> Result<()> {
    // TODO: Implement worker delete logic
    Narration::new(ACTOR_QUEEN_ROUTER, "worker_delete", &job_id)
        .human(format!(
            "TODO: Delete worker {} on hive {} for job {}",
            id, hive_id, job_id
        ))
        .emit();
    Ok(())
}

// ============================================================================
// MODEL OPERATION HANDLERS
// ============================================================================

async fn handle_model_download_job(
    _state: JobRouterState,
    job_id: String,
    hive_id: String,
    model: String,
) -> Result<()> {
    // TODO: Implement model download logic
    Narration::new(ACTOR_QUEEN_ROUTER, "model_download", &job_id)
        .human(format!(
            "TODO: Download model {} on hive {} for job {}",
            model, hive_id, job_id
        ))
        .emit();
    Ok(())
}

async fn handle_model_list_job(
    _state: JobRouterState,
    job_id: String,
    hive_id: String,
) -> Result<()> {
    // TODO: Implement model list logic
    Narration::new(ACTOR_QUEEN_ROUTER, "model_list", &job_id)
        .human(format!("TODO: List models on hive {} for job {}", hive_id, job_id))
        .emit();
    Ok(())
}

async fn handle_model_get_job(
    _state: JobRouterState,
    job_id: String,
    hive_id: String,
    id: String,
) -> Result<()> {
    // TODO: Implement model get logic
    Narration::new(ACTOR_QUEEN_ROUTER, "model_get", &job_id)
        .human(format!(
            "TODO: Get model {} on hive {} for job {}",
            id, hive_id, job_id
        ))
        .emit();
    Ok(())
}

async fn handle_model_delete_job(
    _state: JobRouterState,
    job_id: String,
    hive_id: String,
    id: String,
) -> Result<()> {
    // TODO: Implement model delete logic
    Narration::new(ACTOR_QUEEN_ROUTER, "model_delete", &job_id)
        .human(format!(
            "TODO: Delete model {} on hive {} for job {}",
            id, hive_id, job_id
        ))
        .emit();
    Ok(())
}

// ============================================================================
// INFERENCE OPERATION HANDLER
// ============================================================================

#[allow(clippy::too_many_arguments)]
async fn handle_infer_job(
    _state: JobRouterState,
    job_id: String,
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
    Narration::new(ACTOR_QUEEN_ROUTER, "infer", &job_id)
        .human(format!(
            "TODO: Run inference on hive {} with model {} (prompt: '{}', max_tokens: {}, temp: {}) for job {}",
            hive_id, model, prompt, max_tokens, temperature, job_id
        ))
        .emit();
    Ok(())
}
