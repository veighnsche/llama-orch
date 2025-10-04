//! POST /execute endpoint - Execute inference
//!
//! This endpoint accepts inference requests, validates parameters,
//! and returns an SSE stream of tokens.
//!
//! # Spec References
//! - M0-W-1300: Inference endpoint
//! - M0-W-1302: Request validation

use crate::http::validation::{ExecuteRequest, ValidationError};
use axum::{
    response::{sse::Event, Sse},
    Json,
};
use futures::stream::{self, Stream, StreamExt};
use observability_narration_core::{Narration, ACTION_INFERENCE_START, ACTOR_WORKER_ORCD};
use serde::Serialize;
use std::convert::Infallible;
use tracing::{debug, info};

#[derive(Serialize)]
struct StartedEvent {
    job_id: String,
    started_at: String,
}

/// SSE event: Token generated
#[derive(Serialize)]
struct TokenEvent {
    /// Token text
    t: String,
    /// Token index
    i: u32,
}

/// SSE event: Inference complete
#[derive(Serialize)]
struct EndEvent {
    /// Total tokens generated
    tokens_out: u32,
}

/// Handle POST /execute
///
/// Accepts an inference request, validates parameters, and returns
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
///
/// # Errors
/// Returns 400 Bad Request if validation fails, with details about
/// which field failed and why.
#[axum::debug_handler]
pub async fn handle_execute(
    Json(req): Json<ExecuteRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ValidationError> {
    // Validate request
    req.validate()?;

    info!(
        job_id = %req.job_id,
        prompt_len = req.prompt.len(),
        max_tokens = req.max_tokens,
        temperature = req.temperature,
        "Inference request received"
    );

    // Narrate inference start
    Narration::new(ACTOR_WORKER_ORCD, ACTION_INFERENCE_START, &req.job_id)
        .human(format!("Starting inference for job {}", req.job_id))
        .job_id(&req.job_id)
        .emit();

    debug!(job_id = %req.job_id, "Opening SSE stream");

    // Create placeholder SSE stream
    // This will be replaced with real CUDA inference in FT-006
    let job_id = req.job_id.clone();
    let stream = stream::iter(vec![
        // Event 1: started
        Event::default()
            .event("started")
            .json_data(&StartedEvent {
                job_id: job_id.clone(),
                started_at: chrono::Utc::now().to_rfc3339(),
            })
            .ok(),
        // Event 2: token (placeholder)
        Event::default().event("token").json_data(&TokenEvent { t: "test".to_string(), i: 0 }).ok(),
        // Event 3: end
        Event::default().event("end").json_data(&EndEvent { tokens_out: 1 }).ok(),
    ])
    .filter_map(|event| async move { event })
    .map(Ok);

    Ok(Sse::new(stream))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_started_event_serialization() {
        let event = StartedEvent {
            job_id: "test-123".to_string(),
            started_at: "2025-10-04T12:00:00Z".to_string(),
        };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("test-123"));
        assert!(json.contains("started_at"));
    }

    #[test]
    fn test_token_event_serialization() {
        let event = TokenEvent { t: "hello".to_string(), i: 5 };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("hello"));
        assert!(json.contains("\"i\":5"));
    }

    #[test]
    fn test_end_event_serialization() {
        let event = EndEvent { tokens_out: 42 };

        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("tokens_out"));
        assert!(json.contains("42"));
    }
}

// ---
// Built by Foundation-Alpha 🏗️
