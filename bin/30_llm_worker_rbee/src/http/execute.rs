//! POST /v1/inference endpoint - Execute inference
//!
//! Modified by: TEAM-017 (updated to use Mutex-wrapped backend)
//! Modified by: TEAM-035 (renamed to /v1/inference, added [DONE] marker)
//! Modified by: TEAM-039 (added narration channel for real-time user visibility)
//! Modified by: TEAM-149 (real-time streaming with request queue)
//! Modified by: TEAM-150 (fixed streaming hang - removed blocking narration_stream)

use crate::backend::request_queue::{GenerationRequest, RequestQueue, TokenResponse};
use crate::common::SamplingConfig;
use crate::http::{
    // TEAM-150: Removed narration_channel - it was blocking the stream
    sse::InferenceEvent,
    validation::{ExecuteRequest, ValidationErrorResponse},
};
use crate::narration::{
    self, ACTION_ERROR, ACTION_EXECUTE_REQUEST, ACTOR_HTTP_SERVER,
};
use axum::{
    extract::State,
    response::{sse::Event, Sse},
    Json,
};
use futures::future;
use futures::stream::{self, Stream, StreamExt};
use observability_narration_core::NarrationFields;
use std::{convert::Infallible, sync::Arc};
// TEAM-150: Removed UnboundedReceiverStream - no longer needed
use tracing::{info, warn};

type EventStream = Box<dyn Stream<Item = Result<Event, Infallible>> + Send + Unpin>;

/// Handle POST /v1/inference
///
/// Streams inference results via SSE with OpenAI-compatible [DONE] marker.
///
/// TEAM-017: Updated to use Mutex-wrapped backend for &mut self
/// TEAM-035: Added [DONE] marker for `OpenAI` compatibility
/// TEAM-039: Added narration channel for real-time user visibility
/// TEAM-149: Real-time streaming - tokens sent as generated, not after completion
pub async fn handle_execute(
    State(queue): State<Arc<RequestQueue>>,
    Json(req): Json<ExecuteRequest>,
) -> Result<Sse<EventStream>, ValidationErrorResponse> {
    // TEAM-150: Removed narration channel - it was blocking the stream
    // let narration_rx = narration_channel::create_channel();

    // Validate request
    if let Err(validation_errors) = req.validate_all() {
        warn!(job_id = %req.job_id, "Validation failed");

        narration::narrate_dual(NarrationFields {
            actor: ACTOR_HTTP_SERVER,
            action: ACTION_ERROR,
            target: req.job_id.clone(),
            human: format!("Validation failed for job {}", req.job_id),
            cute: Some(format!("Job {} has invalid parameters! 😟", req.job_id)),
            error_kind: Some("validation_failed".to_string()),
            job_id: Some(req.job_id.clone()),
            ..Default::default()
        });

        return Err(validation_errors);
    }

    info!(job_id = %req.job_id, "Inference request validated");

    narration::narrate_dual(NarrationFields {
        actor: ACTOR_HTTP_SERVER,
        action: ACTION_EXECUTE_REQUEST,
        target: req.job_id.clone(),
        human: format!("Inference request validated for job {}", req.job_id),
        cute: Some(format!("Job {} looks good, let's go! ✅", req.job_id)),
        job_id: Some(req.job_id.clone()),
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

    // TEAM-149: Create channel for this request
    // Tokens will flow through this channel from generation engine to SSE stream
    let (response_tx, mut response_rx) = tokio::sync::mpsc::unbounded_channel();
    
    // TEAM-149: Add request to queue
    // Generation happens in spawn_blocking, HTTP handler returns immediately
    let generation_request = GenerationRequest {
        request_id: req.job_id.clone(),
        prompt: req.prompt.clone(),
        config,
        response_tx,
    };
    
    if let Err(e) = queue.add_request(generation_request) {
        warn!(job_id = %req.job_id, error = %e, "Failed to queue request");

        narration::narrate_dual(NarrationFields {
            actor: ACTOR_HTTP_SERVER,
            action: ACTION_ERROR,
            target: req.job_id.clone(),
            human: format!("Failed to queue request for job {}: {}", req.job_id, e),
            cute: Some(format!("Oh no! Couldn't queue job {}: {} 😟", req.job_id, e)),
            error_kind: Some("queue_failed".to_string()),
            job_id: Some(req.job_id.clone()),
            ..Default::default()
        });

        let events = vec![InferenceEvent::Error {
            code: "QUEUE_FAILED".to_string(),
            message: e,
        }];
        let stream: EventStream = Box::new(
            stream::iter(events)
                .map(|event| Ok(Event::default().json_data(&event).unwrap()))
                .chain(stream::once(future::ready(Ok(Event::default().data("[DONE]"))))),
        );
        return Ok(Sse::new(stream));
    }
    
    // TEAM-149: Return stream immediately!
    // Generation happens in background, tokens flow through response_rx
    let started_event = InferenceEvent::Started {
        job_id: req.job_id.clone(),
        model: "model".to_string(),
        started_at: chrono::Utc::now().timestamp().to_string(),
    };
    
    // TEAM-149: Convert token responses to SSE events
    // Tokens arrive in REAL-TIME as they're generated!
    let mut token_count = 0u32;
    let token_events = Box::pin(async_stream::stream! {
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

    // TEAM-150: Build the complete SSE stream
    // CRITICAL FIX: Removed narration_stream - it was blocking because:
    // 1. Sender stored in thread-local storage never dropped
    // 2. Generation engine runs in spawn_blocking (different thread)
    // 3. Stream waited forever for messages that never came
    // 4. token_events never started because narration_stream blocked!
    //
    // Stream now: Started event → Token events → [DONE]
    let started_stream = stream::once(future::ready(
        Ok(Event::default().json_data(&started_event).unwrap())
    ));

    let stream_with_done: EventStream = Box::new(
        started_stream
            .chain(token_events)
            .chain(stream::once(future::ready(Ok(Event::default().data("[DONE]"))))),
    );

    info!(job_id = %req.job_id, "Streaming inference started");

    // TEAM-149: CRITICAL - We return immediately here!
    // - HTTP handler returns in <100ms
    // - Generation happens in spawn_blocking
    // - Tokens stream in REAL-TIME as they're generated
    // - No more blocking the async runtime!
    Ok(Sse::new(stream_with_done))
}
