//! Job creation HTTP endpoint
//!
//! TEAM-186: Job-based architecture - ALL operations go through POST /v1/jobs

use axum::{extract::State, http::StatusCode, Json};
use job_registry::JobRegistry;
use observability_narration_core::Narration;
use queen_rbee_hive_catalog::HiveCatalog;
use serde::Serialize;
use std::sync::Arc;

const ACTOR_QUEEN_HTTP: &str = "👑 queen-http";
const ACTION_JOB_CREATE: &str = "job_create";

#[derive(Debug, Serialize)]
pub struct HttpJobResponse {
    pub job_id: String,
    pub sse_url: String,
}

#[derive(Clone)]
pub struct SchedulerState {
    pub registry: Arc<JobRegistry<String>>,
    pub hive_catalog: Arc<HiveCatalog>,
}

/// POST /v1/jobs - Create a new job (ALL operations)
///
/// TEAM-186: Generic job endpoint that routes based on "operation" field
/// 
/// **Job Lifecycle:**
/// 1. POST /v1/jobs - Creates job, stores payload, returns job_id + sse_url
/// 2. GET /v1/jobs/{job_id}/stream - Client connects, job starts executing
///
/// This allows the client to connect to the SSE stream before execution starts,
/// ensuring no events are missed.
///
/// Example payloads:
/// - {"operation": "hive_list"}
/// - {"operation": "worker_spawn", "hive_id": "localhost", "model": "...", "worker": "cpu", "device": 0}
/// - {"operation": "infer", "hive_id": "localhost", "model": "...", "prompt": "...", ...}
pub async fn handle_create_job(
    State(state): State<SchedulerState>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<HttpJobResponse>, (StatusCode, String)> {
    // TEAM-186: Create job in registry but DON'T execute yet
    let job_id = state.registry.create_job();
    let sse_url = format!("/v1/jobs/{}/stream", job_id);

    // TEAM-186: Store payload in job registry for later execution
    // Job will execute when client connects to SSE stream
    state.registry.set_payload(&job_id, payload);

    Narration::new(ACTOR_QUEEN_HTTP, ACTION_JOB_CREATE, &job_id)
        .human(format!("Job {} created, waiting for client connection", job_id))
        .emit();

    Ok(Json(HttpJobResponse { job_id, sse_url }))
}
