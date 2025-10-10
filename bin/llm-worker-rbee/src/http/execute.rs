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
use crate::narration::{self, ACTOR_HTTP_SERVER, ACTION_ERROR, ACTION_EXECUTE_REQUEST, ACTOR_CANDLE_BACKEND};
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

    // TEAM-017: Execute inference with mutex lock
    let result = match backend.lock().await.execute(&req.prompt, &config).await {
        Ok(r) => r,
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

    info!(job_id = %req.job_id, tokens = result.tokens.len(), "Inference complete");

    // Convert result to SSE events
    let mut events = vec![InferenceEvent::Started {
        job_id: req.job_id.clone(),
        model: "model".to_string(),
        started_at: "0".to_string(),
    }];

    for (i, token) in result.tokens.iter().enumerate() {
        events.push(InferenceEvent::Token { t: token.clone(), i: i as u32 });
    }

    events.push(InferenceEvent::End {
        tokens_out: result.tokens.len() as u32,
        decode_time_ms: result.decode_time_ms,
        stop_reason: result.stop_reason,
        stop_sequence_matched: result.stop_sequence_matched,
    });

    // TEAM-039: Merge narration events with token events
    // Convert narration receiver to stream
    let narration_stream = UnboundedReceiverStream::new(narration_rx);

    // Convert token events to stream
    let token_stream = stream::iter(events);

    // Interleave narration and token events
    // Note: Since inference is synchronous, narration events will be buffered
    // and emitted before token events. For true real-time streaming, we'd need
    // a streaming backend.
    let merged_stream = narration_stream
        .chain(token_stream)
        .map(|event| Ok(Event::default().json_data(&event).unwrap()));

    // TEAM-035: Add [DONE] marker after all events (OpenAI compatible)
    let stream_with_done: EventStream = Box::new(
        merged_stream.chain(stream::once(future::ready(Ok(Event::default().data("[DONE]"))))),
    );

    // TEAM-039: Clean up narration channel when stream is dropped
    // Note: The channel will be cleaned up when the receiver (narration_rx) is dropped
    // which happens when the stream completes. No explicit cleanup needed.

    Ok(Sse::new(stream_with_done))
}
