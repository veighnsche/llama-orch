//! POST /execute endpoint - Execute inference
//!
//! This endpoint accepts inference requests, validates parameters,
//! and returns an SSE stream of tokens.
//!
//! - M0-W-1300: Inference endpoint
//! - M0-W-1302: Request validation
//! - M0-W-1310: SSE streaming

use crate::http::sse::{InferenceEvent, StopReason};
use crate::http::validation::{ExecuteRequest, ValidationErrorResponse};
use axum::{
    extract::Extension,
    response::{sse::Event, Sse},
    Json,
};
use futures::stream::{self, Stream, StreamExt};
use observability_narration_core::{Narration, ACTION_INFERENCE_START, ACTION_INFERENCE_COMPLETE, ACTOR_WORKER_ORCD};
use std::convert::Infallible;
use tracing::{debug, info, warn};

/// Handle POST /execute
///
/// a placeholder SSE stream. This is a skeleton implementation that
/// will be wired to CUDA inference in a later story.
/// # Request Format (Basic - Sprint 3)
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
/// # Request Format (Advanced - Sprint 4)
/// ```json
/// {
///   "job_id": "unique-job-id",
///   "prompt": "Your prompt text here",
///   "max_tokens": 100,
///   "temperature": 0.7,
///   "seed": 42,
///   "top_p": 0.9,
///   "top_k": 50,
///   "repetition_penalty": 1.1,
///   "stop": ["\\n\\n", "END"],
///   "min_p": 0.05
/// }
/// ```
///
/// # Response Format
/// SSE stream with events:
/// - `started`: Inference began
/// - `token`: Each generated token
/// - `end`: Inference complete (includes stop_reason)
///
/// # Stop Reasons
/// - `max_tokens`: Reached max_tokens limit
/// - `stop_sequence`: Matched a stop sequence (includes stop_sequence_matched)
/// - `cancelled`: Request cancelled by client
/// - `error`: Inference error occurred
///
/// # Errors
/// Returns 400 Bad Request if validation fails, with details about
/// which field failed and why.
#[axum::debug_handler]
pub async fn handle_execute(
    Extension(correlation_id): Extension<String>,
    Json(req): Json<ExecuteRequest>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ValidationErrorResponse> {
    // Validate request (collect all errors)
    if let Err(validation_errors) = req.validate_all() {
        let error_count = validation_errors.errors.len();
        let field_list: Vec<_> =
            validation_errors.errors.iter().map(|e| e.field.as_str()).collect();

        warn!(
            correlation_id = %correlation_id,
            job_id = %req.job_id,
            error_count = error_count,
            fields = ?field_list,
            "Validation failed"
        );

        // Narrate validation failure
        Narration::new(ACTOR_WORKER_ORCD, "validation", &req.job_id)
            .human(format!(
                "Validation failed for job {}: {} errors ({})",
                req.job_id,
                error_count,
                field_list.join(", ")
            ))
            .cute(format!(
                "Oh no! Job {} has {} validation boo-boos in {}! Let's fix them! 😟🔍",
                req.job_id,
                error_count,
                field_list.join(", ")
            ))
            .correlation_id(&correlation_id)
            .job_id(&req.job_id)
            .error_kind("ValidationFailed")
            .emit_warn();

        return Err(validation_errors);
    }

    info!(
        correlation_id = %correlation_id,
        job_id = %req.job_id,
        prompt_len = req.prompt.len(),
        max_tokens = req.max_tokens,
        temperature = req.temperature,
        "Inference request validated and received"
    );

    // Narrate inference start
    Narration::new(ACTOR_WORKER_ORCD, ACTION_INFERENCE_START, &req.job_id)
        .human(format!("Starting inference for job {} ({} tokens max, temp={})", req.job_id, req.max_tokens, req.temperature))
        .cute(format!("Worker gets ready to help with job {}! Time to generate some tokens! 🎯✨", req.job_id))
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
        InferenceEvent::End {
            tokens_out: 1,
            decode_time_ms: 100,
            stop_reason: StopReason::MaxTokens,
            stop_sequence_matched: None,
        },
    ];

    let correlation_id_clone = correlation_id.clone();
    let job_id_clone = job_id.clone();
    
    let stream = stream::iter(events)
        .map(move |event| {
            // Narrate completion when we see the End event
            if matches!(event, InferenceEvent::End { .. }) {
                if let InferenceEvent::End { tokens_out, decode_time_ms, .. } = &event {
                    Narration::new(ACTOR_WORKER_ORCD, ACTION_INFERENCE_COMPLETE, &job_id_clone)
                        .human(format!("Completed inference for job {} ({} tokens in {} ms)", job_id_clone, tokens_out, decode_time_ms))
                        .cute(format!("All done with job {}! Generated {} tokens! Great work! 🎉✨", job_id_clone, tokens_out))
                        .correlation_id(&correlation_id_clone)
                        .job_id(&job_id_clone)
                        .tokens_out(*tokens_out as u64)
                        .decode_time_ms(*decode_time_ms as u64)
                        .emit();
                }
            }
            
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
        let end = InferenceEvent::End {
            tokens_out: 42,
            decode_time_ms: 1000,
            stop_reason: StopReason::MaxTokens,
            stop_sequence_matched: None,
        };

        assert_eq!(end.event_name(), "end");
        assert!(end.is_terminal());
    }

    #[test]
    fn test_end_event_with_stop_sequence() {
        let end = InferenceEvent::End {
            tokens_out: 20,
            decode_time_ms: 500,
            stop_reason: StopReason::StopSequence,
            stop_sequence_matched: Some("\n\n".to_string()),
        };

        assert_eq!(end.event_name(), "end");
        assert!(end.is_terminal());

        // Verify serialization includes stop_sequence_matched
        let json = serde_json::to_string(&end).unwrap();
        assert!(json.contains("stop_sequence"));
        assert!(json.contains("\\n\\n"));
    }

    #[test]
    fn test_end_event_max_tokens_no_stop_sequence() {
        let end = InferenceEvent::End {
            tokens_out: 100,
            decode_time_ms: 2000,
            stop_reason: StopReason::MaxTokens,
            stop_sequence_matched: None,
        };

        // Verify serialization omits stop_sequence_matched when None
        let json = serde_json::to_string(&end).unwrap();
        assert!(json.contains("max_tokens"));
        assert!(!json.contains("stop_sequence_matched"));
    }
}

// ---
// Built by Foundation-Alpha 🏗️
