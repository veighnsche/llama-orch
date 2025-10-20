//! POST /v1/inference endpoint - Create inference job
//!
//! Modified by: TEAM-017 (updated to use Mutex-wrapped backend)
//! Modified by: TEAM-035 (renamed to /v1/inference, added [DONE] marker)
//! Modified by: TEAM-039 (added narration channel for real-time user visibility)
//! Modified by: TEAM-149 (real-time streaming with request queue)
//! Modified by: TEAM-150 (fixed streaming hang - removed blocking narration_stream)
//! Modified by: TEAM-154 (dual-call pattern - POST returns JSON, not SSE)

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

/// Handle POST /v1/inference - Create job and return job_id + sse_url
///
/// TEAM-154: Changed from direct SSE to dual-call pattern
/// - Server generates job_id (client doesn't provide it)
/// - Returns JSON with job_id and sse_url
/// - Client then calls GET /v1/inference/{job_id}/stream for SSE
///
/// TEAM-017: Updated to use Mutex-wrapped backend for &mut self
/// TEAM-035: Added [DONE] marker for `OpenAI` compatibility
/// TEAM-039: Added narration channel for real-time user visibility
/// TEAM-149: Real-time streaming - tokens sent as generated, not after completion
pub async fn handle_create_job(
    State(state): State<WorkerState>,
    Json(req): Json<ExecuteRequest>,
) -> Result<Json<CreateJobResponse>, ValidationErrorResponse> {
    // Validate request
    if let Err(validation_errors) = req.validate_all() {
        warn!("Validation failed");

        narration::narrate_dual(NarrationFields {
            actor: ACTOR_HTTP_SERVER,
            action: ACTION_ERROR,
            target: "validation".to_string(),
            human: "Validation failed for inference request".to_string(),
            cute: Some("Request has invalid parameters! 😟".to_string()),
            error_kind: Some("validation_failed".to_string()),
            ..Default::default()
        });

        return Err(validation_errors);
    }

    // TEAM-154: Create job in registry (generates job_id)
    let job_id = state.registry.create_job();

    info!(job_id = %job_id, "Inference job created");

    narration::narrate_dual(NarrationFields {
        actor: ACTOR_HTTP_SERVER,
        action: ACTION_EXECUTE_REQUEST,
        target: job_id.clone(),
        human: format!("Inference job {} created and validated", job_id),
        cute: Some(format!("Job {} looks good, let's go! ✅", job_id)),
        job_id: Some(job_id.clone()),
        ..Default::default()
    });

    // Convert request to sampling config
    let config = SamplingConfig {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        repetition_penalty: req.repetition_penalty,
        min_p: req.min_p,
        stop_sequences: vec![],
        stop_strings: req.stop.clone(),
        seed: req.seed.unwrap_or(42),
        max_tokens: req.max_tokens,
    };

    // TEAM-154 FIX: Create channel for this job
    // - Sender goes to generation engine (via GenerationRequest)
    // - Receiver stored in registry for GET endpoint to consume
    let (response_tx, response_rx) = tokio::sync::mpsc::unbounded_channel();
    state.registry.set_token_receiver(&job_id, response_rx);

    // TEAM-154: Add request to queue
    // Generation happens in spawn_blocking, HTTP handler returns immediately
    let generation_request = GenerationRequest {
        request_id: job_id.clone(),
        prompt: req.prompt.clone(),
        config,
        response_tx,
    };

    if let Err(e) = state.queue.add_request(generation_request) {
        warn!(job_id = %job_id, error = %e, "Failed to queue request");

        narration::narrate_dual(NarrationFields {
            actor: ACTOR_HTTP_SERVER,
            action: ACTION_ERROR,
            target: job_id.clone(),
            human: format!("Failed to queue request for job {}: {}", job_id, e),
            cute: Some(format!("Oh no! Couldn't queue job {}: {} 😟", job_id, e)),
            error_kind: Some("queue_failed".to_string()),
            job_id: Some(job_id.clone()),
            ..Default::default()
        });

        return Err(ValidationErrorResponse::single_error("queue", "Failed to queue job"));
    }

    // TEAM-154: Return job_id and sse_url (JSON response, not SSE!)
    // Client will then call GET /v1/inference/{job_id}/stream for SSE
    let sse_url = format!("/v1/inference/{}/stream", job_id);

    info!(job_id = %job_id, sse_url = %sse_url, "Job created, returning SSE URL");

    narration::narrate_dual(NarrationFields {
        actor: ACTOR_HTTP_SERVER,
        action: "job_created",
        target: job_id.clone(),
        human: format!("Job {} created, SSE URL: {}", job_id, sse_url),
        cute: Some(format!("Job {} is ready! Stream at {} 🎉", job_id, sse_url)),
        job_id: Some(job_id.clone()),
        ..Default::default()
    });

    // TEAM-154: Return JSON response (not SSE stream!)
    Ok(Json(CreateJobResponse { job_id, sse_url }))
}
