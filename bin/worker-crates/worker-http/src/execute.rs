//! POST /execute endpoint - Execute inference

use crate::{
    backend::InferenceBackend,
    sse::InferenceEvent,
    validation::{ExecuteRequest, ValidationErrorResponse},
};
use axum::{
    extract::State,
    response::{sse::Event, Sse},
    Json,
};
use futures::stream::{self, Stream, StreamExt};
use std::{convert::Infallible, sync::Arc};
use tracing::{info, warn};
use worker_common::SamplingConfig;

type EventStream = Box<dyn Stream<Item = Result<Event, Infallible>> + Send + Unpin>;

/// Handle POST /execute
pub async fn handle_execute<B: InferenceBackend>(
    State(backend): State<Arc<B>>,
    Json(req): Json<ExecuteRequest>,
) -> Result<Sse<EventStream>, ValidationErrorResponse> {
    // Validate request
    if let Err(validation_errors) = req.validate_all() {
        warn!(job_id = %req.job_id, "Validation failed");
        return Err(validation_errors);
    }

    info!(job_id = %req.job_id, "Inference request validated");

    // Convert request to sampling config
    let config = SamplingConfig::from_request(&req);

    // Execute inference
    let result = match backend.execute(&req.prompt, &config).await {
        Ok(r) => r,
        Err(e) => {
            warn!(job_id = %req.job_id, error = %e, "Inference failed");
            let events = vec![InferenceEvent::Error {
                code: "INFERENCE_FAILED".to_string(),
                message: e.to_string(),
            }];
            let stream: EventStream = Box::new(stream::iter(events).map(|event| {
                Ok(Event::default().json_data(&event).unwrap())
            }));
            return Ok(Sse::new(stream));
        }
    };

    info!(job_id = %req.job_id, tokens = result.tokens.len(), "Inference complete");

    // Convert result to SSE events
    let mut events = vec![InferenceEvent::Started {
        job_id: req.job_id.clone(),
        model: "model".to_string(),
    }];

    for (i, token) in result.tokens.iter().enumerate() {
        events.push(InferenceEvent::Token {
            t: token.clone(),
            i: i as u32,
        });
    }

    events.push(InferenceEvent::End {
        tokens_out: result.tokens.len() as u32,
        decode_time_ms: result.decode_time_ms,
        stop_reason: result.stop_reason,
        stop_sequence_matched: result.stop_sequence_matched,
    });

    let stream: EventStream = Box::new(stream::iter(events).map(|event| {
        Ok(Event::default().json_data(&event).unwrap())
    }));

    Ok(Sse::new(stream))
}
