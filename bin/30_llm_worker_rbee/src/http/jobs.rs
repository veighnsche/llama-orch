//! POST /v1/jobs endpoint - Create job (job-based architecture)
//!
//! Modified by: TEAM-017 (updated to use Mutex-wrapped backend)
//! Modified by: TEAM-035 (renamed to /v1/inference, added [DONE] marker)
//! Modified by: TEAM-039 (added narration channel for real-time user visibility)
//! Modified by: TEAM-149 (real-time streaming with request queue)
//! Modified by: TEAM-150 (fixed streaming hang - removed blocking `narration_stream`)
//! Modified by: TEAM-154 (dual-call pattern - POST returns JSON, not SSE)
//! Modified by: TEAM-353 (job-based architecture with operations-contract)

use crate::backend::request_queue::GenerationRequest;
use crate::common::SamplingConfig;
use crate::http::routes::WorkerState;
use crate::http::validation::{ExecuteRequest, ValidationErrorResponse};
use crate::narration::{self, ACTION_ERROR, ACTION_EXECUTE_REQUEST, ACTOR_HTTP_SERVER};
use axum::{extract::State, Json};
use observability_narration_core::NarrationFields;
use serde::Serialize;
use tracing::{info, warn};

/// Response from creating a job
///
/// TEAM-154: Dual-call pattern response
#[derive(Debug, Serialize)]
pub struct CreateJobResponse {
    pub job_id: String,
    pub sse_url: String,
}

/// Handle POST /v1/jobs - Create job from Operation (job-based architecture)
///
/// TEAM-353: Thin HTTP wrapper that delegates to job_router::create_job()
/// Accepts Operation enum from operations-contract (mirrors Hive/Queen pattern)
///
/// TEAM-154: Changed from direct SSE to dual-call pattern
/// - Server generates `job_id` (client doesn't provide it)
/// - Returns JSON with `job_id` and `sse_url`
/// - Client then calls GET /v1/jobs/{job_id}/stream for SSE
pub async fn handle_create_job(
    State(state): State<WorkerState>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<crate::job_router::JobResponse>, (axum::http::StatusCode, String)> {
    // TEAM-353: Delegate to job_router (mirrors Hive pattern)
    let job_state = crate::job_router::JobState {
        registry: state.registry,
        queue: state.queue,
    };

    crate::job_router::create_job(job_state, payload)
        .await
        .map(Json)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}
