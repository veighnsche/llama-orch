//! POST /v1/inference endpoint - Execute inference
//!
//! Modified by: TEAM-017 (updated to use Mutex-wrapped backend)
//! Modified by: TEAM-035 (renamed to /v1/inference, added [DONE] marker)
//! Modified by: TEAM-039 (added narration channel for real-time user visibility)

use crate::common::SamplingConfig;
use crate::http::{
    backend::InferenceBackend,
    narration_channel,
    sse::InferenceEvent,
    validation::{ExecuteRequest, ValidationErrorResponse},
};
use crate::narration::{
    self, ACTION_ERROR, ACTION_EXECUTE_REQUEST, ACTOR_CANDLE_BACKEND, ACTOR_HTTP_SERVER,
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
use tokio::sync::Mutex;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tracing::{info, warn};

type EventStream = Box<dyn Stream<Item = Result<Event, Infallible>> + Send + Unpin>;

/// Handle POST /v1/inference
///
/// Streams inference results via SSE with OpenAI-compatible [DONE] marker.
///
/// TEAM-017: Updated to use Mutex-wrapped backend for &mut self
/// TEAM-035: Added [DONE] marker for `OpenAI` compatibility
/// TEAM-039: Added narration channel for real-time user visibility
pub async fn handle_execute<B: InferenceBackend>(
    State(backend): State<Arc<Mutex<B>>>,
    Json(req): Json<ExecuteRequest>,
) -> Result<Sse<EventStream>, ValidationErrorResponse> {
    // TEAM-039: Create narration channel for this request
    let narration_rx = narration_channel::create_channel();

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

    // TEAM-147: Use REAL streaming backend (tokens yielded as generated)
    let token_stream = match backend.lock().await.execute_stream(&req.prompt, &config).await {
        Ok(s) => s,
        Err(e) => {
            warn!(job_id = %req.job_id, error = %e, "Inference failed");

            narration::narrate_dual(NarrationFields {
                actor: ACTOR_CANDLE_BACKEND,
                action: ACTION_ERROR,
                target: req.job_id.clone(),
                human: format!("Inference failed for job {}: {}", req.job_id, e),
                cute: Some(format!("Oh no! Job {} hit a snag: {} 😟", req.job_id, e)),
                error_kind: Some("inference_failed".to_string()),
                job_id: Some(req.job_id.clone()),
                ..Default::default()
            });

            let events = vec![InferenceEvent::Error {
                code: "INFERENCE_FAILED".to_string(),
                message: e.to_string(),
            }];
            // TEAM-035: Add [DONE] marker after error (OpenAI compatible)
            let stream: EventStream = Box::new(
                stream::iter(events)
                    .map(|event| Ok(Event::default().json_data(&event).unwrap()))
                    .chain(stream::once(future::ready(Ok(Event::default().data("[DONE]"))))),
            );
            return Ok(Sse::new(stream));
        }
    };

    // TEAM-147: Stream tokens in REAL-TIME as they're generated
    // Send "started" event first
    let started_event = InferenceEvent::Started {
        job_id: req.job_id.clone(),
        model: "model".to_string(),
        started_at: chrono::Utc::now().timestamp().to_string(),
    };

    // TEAM-147: Convert token stream to SSE events
    // Tokens are yielded as they're generated, not after completion!
    let mut token_count = 0u32;
    let token_events = token_stream.map(move |token_result| {
        match token_result {
            Ok(token) => {
                let event = InferenceEvent::Token { 
                    t: token, 
                    i: token_count 
                };
                token_count += 1;
                Ok(Event::default().json_data(&event).unwrap())
            }
            Err(e) => {
                let event = InferenceEvent::Error {
                    code: "TOKEN_ERROR".to_string(),
                    message: e.to_string(),
                };
                Ok(Event::default().json_data(&event).unwrap())
            }
        }
    });

    // TEAM-147: Build the complete SSE stream
    // 1. Narration events (buffered from start)
    // 2. Started event
    // 3. Token events (REAL-TIME streaming!)
    // 4. [DONE] marker
    let narration_stream = UnboundedReceiverStream::new(narration_rx)
        .map(|event| Ok(Event::default().json_data(&event).unwrap()));

    let started_stream = stream::once(future::ready(
        Ok(Event::default().json_data(&started_event).unwrap())
    ));

    let stream_with_done: EventStream = Box::new(
        narration_stream
            .chain(started_stream)
            .chain(token_events)
            .chain(stream::once(future::ready(Ok(Event::default().data("[DONE]"))))),
    );

    info!(job_id = %req.job_id, "Streaming inference started");

    // TEAM-147: Tokens now stream in REAL-TIME as they're generated!
    // No more waiting for completion before streaming.
    Ok(Sse::new(stream_with_done))
}
