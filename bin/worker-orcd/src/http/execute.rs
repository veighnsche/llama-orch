//! POST /execute endpoint - Execute inference
//!
//! This endpoint accepts inference requests, validates parameters,
//! and returns an SSE stream of tokens.
//!
//! # Spec References
//! - M0-W-1300: Inference endpoint
//! - M0-W-1302: Request validation
//! - M0-W-1310: SSE streaming

use crate::http::sse::InferenceEvent;
use crate::http::validation::{ExecuteRequest, ValidationError};
use axum::{
    extract::Extension,
    response::{sse::Event, Sse},
    Json,
};
use futures::stream::{self, Stream, StreamExt};
use observability_narration_core::{Narration, ACTION_INFERENCE_START, ACTOR_WORKER_ORCD};
use std::convert::Infallible;
use tracing::{debug, info};

/// Handle POST /execute
///
/// a placeholder SSE stream. This is a skeleton implementation that
/// will be wired to CUDA inference in a later story.
///
/// # Request Format
/// ```json
/// {
///   "job_id": "unique-job-id",
///   "prompt": "Your prompt text here",
///   "max_tokens": 100,
///   "temperature": 0.7,
///   "seed": 42
/// }
/// ```
///
/// # Response Format
/// SSE stream with events:
/// - `started`: Inference began
/// - `token`: Each generated token
/// - `end`: Inference complete
/// # Errors
/// Returns 400 Bad Request if validation fails, with details about
/// which field failed and why.
#[axum::debug_handler]
pub async fn handle_execute(
    Extension(correlation_id): Extension<String>,
    Json(req): Json<ExecuteRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ValidationError> {
    // Validate request
    req.validate()?;

    info!(
        correlation_id = %correlation_id,
        job_id = %req.job_id,
        prompt_len = req.prompt.len(),
        max_tokens = req.max_tokens,
        temperature = req.temperature,
        "Inference request received"
    );

    // Narrate inference start
    Narration::new(ACTOR_WORKER_ORCD, ACTION_INFERENCE_START, &req.job_id)
        .human(format!("Starting inference for job {}", req.job_id))
        .correlation_id(&correlation_id)
        .job_id(&req.job_id)
        .emit();

    debug!(
        correlation_id = %correlation_id,
        job_id = %req.job_id,
        "Opening SSE stream"
    );

    // Create placeholder SSE stream using InferenceEvent types
    // This will be replaced with real CUDA inference in FT-006
    let job_id = req.job_id.clone();
    let events = vec![
        InferenceEvent::Started {
            job_id: job_id.clone(),
            model: "placeholder".to_string(),
            started_at: chrono::Utc::now().to_rfc3339(),
        },
        InferenceEvent::Token { t: "test".to_string(), i: 0 },
        InferenceEvent::End { tokens_out: 1, decode_time_ms: 100 },
    ];

    let stream = stream::iter(events)
        .map(|event| {
            let event_name = event.event_name();
            Event::default().event(event_name).json_data(&event)
        })
        .filter_map(|result| async move { result.ok() })
        .map(Ok);

    Ok(Sse::new(stream))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::sse::InferenceEvent;

    #[test]
    fn test_inference_event_integration() {
        // Test that InferenceEvent types work with the execute handler
        let started = InferenceEvent::Started {
            job_id: "test-123".to_string(),
            model: "test-model".to_string(),
            started_at: "2025-10-04T12:00:00Z".to_string(),
        };

        assert_eq!(started.event_name(), "started");
        assert!(!started.is_terminal());
    }

    #[test]
    fn test_token_event_integration() {
        let token = InferenceEvent::Token { t: "hello".to_string(), i: 5 };

        assert_eq!(token.event_name(), "token");
        assert!(!token.is_terminal());
    }

    #[test]
    fn test_end_event_integration() {
        let end = InferenceEvent::End { tokens_out: 42, decode_time_ms: 1000 };

        assert_eq!(end.event_name(), "end");
        assert!(end.is_terminal());
    }
}

// ---
// Built by Foundation-Alpha 🏗️
