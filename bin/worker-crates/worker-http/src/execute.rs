//! POST /execute endpoint - Execute inference
//!
//! Simplified version using InferenceBackend trait

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
use tracing::{debug, info, warn};
use worker_common::SamplingConfig;

/// Handle POST /execute
pub async fn handle_execute<B: InferenceBackend>(
    State(backend): State<Arc<B>>,
    Json(req): Json<ExecuteRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ValidationErrorResponse> {
    // Validate request
    if let Err(validation_errors) = req.validate_all() {
        warn!(
            job_id = %req.job_id,
            error_count = validation_errors.errors.len(),
            "Validation failed"
        );
        return Err(validation_errors);
    }

    info!(
        job_id = %req.job_id,
        prompt_len = req.prompt.len(),
        max_tokens = req.max_tokens,
        "Inference request validated"
    );

    // Convert request to sampling config
    let config = SamplingConfig::from_request(&req);

    // Execute inference
    let result = match backend.execute(&req.prompt, &config).await {
        Ok(r) => r,
        Err(e) => {
            warn!(job_id = %req.job_id, error = %e, "Inference failed");
            let error_event = InferenceEvent::End {
                tokens_out: 0,
                decode_time_ms: 0,
                stop_reason: worker_common::inference_result::StopReason::Error,
                stop_sequence_matched: None,
            };
            let events = vec![error_event];
            let stream = stream::iter(events).map(|event| {
                Ok(Event::default().json_data(&event).unwrap())
            });
            return Ok(Sse::new(stream));
        }
    };

    debug!(job_id = %req.job_id, tokens = result.tokens.len(), "Inference complete");

    // Convert result to SSE events
    let mut events = vec![InferenceEvent::Started {
        job_id: req.job_id.clone(),
        model: "model".to_string(),
        started_at: "0".to_string(),
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

    let stream = stream::iter(events).map(|event| {
        Ok(Event::default().json_data(&event).unwrap())
    });

    Ok(Sse::new(stream))
}
