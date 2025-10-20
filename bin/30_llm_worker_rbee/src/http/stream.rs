//! GET /v1/inference/{job_id}/stream - Stream job results via SSE
//!
//! Created by: TEAM-154 (dual-call pattern implementation)
//!
//! This endpoint implements the second half of the dual-call pattern:
//! 1. POST /v1/inference creates job and returns job_id + sse_url
//! 2. GET /v1/inference/{job_id}/stream streams results via SSE (this file)

use crate::http::routes::WorkerState;
use crate::http::sse::InferenceEvent;
use crate::narration::{self, ACTION_ERROR, ACTOR_HTTP_SERVER};
use axum::{
    extract::{Path, State},
    response::{sse::Event, Sse},
};
use futures::stream::{self, Stream, StreamExt};
use job_registry::JobState;
use observability_narration_core::NarrationFields;
use std::convert::Infallible;
use tracing::{info, warn};

type EventStream = Box<dyn Stream<Item = Result<Event, Infallible>> + Send + Unpin>;

/// Handle GET /v1/inference/{job_id}/stream
///
/// TEAM-154: New endpoint for dual-call pattern
/// - Retrieves job from registry
/// - Subscribes to token stream
/// - Streams SSE events to client
pub async fn handle_stream_job(
    Path(job_id): Path<String>,
    State(state): State<WorkerState>,
) -> Result<Sse<EventStream>, (axum::http::StatusCode, String)> {
    info!(job_id = %job_id, "Stream request received");

    // Check if job exists
    if !state.registry.has_job(&job_id) {
        warn!(job_id = %job_id, "Job not found");

        narration::narrate_dual(NarrationFields {
            actor: ACTOR_HTTP_SERVER,
            action: ACTION_ERROR,
            target: job_id.clone(),
            human: format!("Job {} not found", job_id),
            cute: Some(format!("Can't find job {}! 😟", job_id)),
            error_kind: Some("job_not_found".to_string()),
            job_id: Some(job_id.clone()),
            ..Default::default()
        });

        return Err((axum::http::StatusCode::NOT_FOUND, format!("Job {} not found", job_id)));
    }

    // Check job state
    if let Some(JobState::Failed(error)) = state.registry.get_job_state(&job_id) {
        warn!(job_id = %job_id, error = %error, "Job failed");

        narration::narrate_dual(NarrationFields {
            actor: ACTOR_HTTP_SERVER,
            action: ACTION_ERROR,
            target: job_id.clone(),
            human: format!("Job {} failed: {}", job_id, error),
            cute: Some(format!("Job {} failed! 😟", job_id)),
            error_kind: Some("job_failed".to_string()),
            job_id: Some(job_id.clone()),
            ..Default::default()
        });

        return Err((
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Job {} failed: {}", job_id, error),
        ));
    }

    // TEAM-154 FIX: Take the receiver from the registry
    // This consumes it - can only be called once per job!
    let mut response_rx = state.registry.take_token_receiver(&job_id).ok_or_else(|| {
        warn!(job_id = %job_id, "Job has no token receiver");

        narration::narrate_dual(NarrationFields {
            actor: ACTOR_HTTP_SERVER,
            action: ACTION_ERROR,
            target: job_id.clone(),
            human: format!("Job {} has no token receiver", job_id),
            cute: Some(format!("Job {} isn't ready yet! 😟", job_id)),
            error_kind: Some("no_token_receiver".to_string()),
            job_id: Some(job_id.clone()),
            ..Default::default()
        });

        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            format!("Job {} has no token receiver", job_id),
        )
    })?;

    narration::narrate_dual(NarrationFields {
        actor: ACTOR_HTTP_SERVER,
        action: "stream_started",
        target: job_id.clone(),
        human: format!("SSE stream started for job {}", job_id),
        cute: Some(format!("Streaming job {} now! 📡", job_id)),
        job_id: Some(job_id.clone()),
        ..Default::default()
    });

    // Build SSE stream
    let started_event = InferenceEvent::Started {
        job_id: job_id.clone(),
        model: "model".to_string(),
        started_at: chrono::Utc::now().timestamp().to_string(),
    };

    // TEAM-154 FIX: Convert token responses to SSE events
    // Tokens arrive in REAL-TIME as they're generated!
    // This is the SAME code that worked in the single-call implementation
    let mut token_count = 0u32;
    let token_events = Box::pin(async_stream::stream! {
        use crate::backend::request_queue::TokenResponse;

        while let Some(token_response) = response_rx.recv().await {
            match token_response {
                TokenResponse::Token(token) => {
                    yield Ok(Event::default().json_data(&InferenceEvent::Token {
                        t: token,
                        i: token_count,
                    }).unwrap());
                    token_count += 1;
                }
                TokenResponse::Error(e) => {
                    yield Ok(Event::default().json_data(&InferenceEvent::Error {
                        code: "GENERATION_ERROR".to_string(),
                        message: e,
                    }).unwrap());
                }
                TokenResponse::Done => {
                    break;
                }
            }
        }
    });

    let started_stream = stream::once(futures::future::ready(Ok(Event::default()
        .json_data(&started_event)
        .unwrap())));

    let stream_with_done: EventStream = Box::new(
        started_stream
            .chain(token_events)
            .chain(stream::once(futures::future::ready(Ok(Event::default().data("[DONE]"))))),
    );

    info!(job_id = %job_id, "Streaming inference started");

    Ok(Sse::new(stream_with_done))
}
