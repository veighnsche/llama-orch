// TEAM-110: Audited 2025-10-18 - ✅ CLEAN - Industry-standard SSE with three-state machine and [DONE] marker
//
//! Model loading progress endpoint
//!
//! Created by: TEAM-035
//!
//! Per test-001-mvp.md Phase 7: Worker Health Check
//! Implements SSE streaming for model loading progress following industry standards
//! from mistral.rs and llama.cpp.
//!
//! # Spec References
//! - `SSE_IMPLEMENTATION_PLAN.md` Phase 2: Model Loading Progress
//! - test-001-mvp.md Lines 243-254: Loading progress events
//!
//! # Industry Standards
//! - [DONE] marker (`OpenAI` compatible)
//! - 10-second keep-alive interval (mistral.rs pattern)
//! - Three-state machine: Running → `SendingDone` → Done
//! - Broadcast channels with 100 buffer size

use axum::{
    extract::State,
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
};
use futures::stream::Stream;
use serde::Serialize;
use std::{convert::Infallible, sync::Arc, time::Duration};
use tokio::sync::Mutex;

use crate::http::backend::InferenceBackend;

/// Loading event states per test-001-mvp.md
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "stage")]
pub enum LoadingEvent {
    /// Model is loading layers into VRAM
    #[serde(rename = "loading_to_vram")]
    LoadingToVram { layers_loaded: u32, layers_total: u32, vram_mb: u64 },
    /// Model is ready for inference
    #[serde(rename = "ready")]
    Ready,
}

/// Stream state machine (mistral.rs pattern)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoadingState {
    Running,     // Actively streaming
    SendingDone, // About to send [DONE]
    Done,        // Completed
}

/// Handle GET /v1/loading/progress
///
/// Streams model loading progress via SSE following industry-standard patterns.
///
/// # Event Format
/// ```json
/// data: {"stage":"loading_to_vram","layers_loaded":12,"layers_total":32,"vram_mb":2048}
/// data: {"stage":"ready"}
/// data: [DONE]
/// ```
///
/// # Industry Standards
/// - OpenAI-compatible [DONE] marker
/// - 10-second keep-alive interval (prevents proxy timeouts)
/// - Three-state machine ensures [DONE] is always sent
/// - Graceful handling of connection drops
///
/// # Errors
/// - 503 `SERVICE_UNAVAILABLE`: Model not currently loading
pub async fn handle_loading_progress<B: InferenceBackend>(
    State(backend): State<Arc<Mutex<B>>>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, String)> {
    // Get loading progress channel from backend
    let mut rx = backend
        .lock()
        .await
        .loading_progress_channel()
        .ok_or((StatusCode::SERVICE_UNAVAILABLE, "Model not loading".to_string()))?;

    // Create SSE stream with industry-standard pattern
    let stream = async_stream::stream! {
        let mut done_state = LoadingState::Running;

        loop {
            match done_state {
                LoadingState::SendingDone => {
                    // Industry standard: Send [DONE] marker (OpenAI compatible)
                    yield Ok(Event::default().data("[DONE]"));
                    done_state = LoadingState::Done;
                }
                LoadingState::Done => {
                    // Stream complete
                    break;
                }
                LoadingState::Running => {
                    match rx.recv().await {
                        Ok(event) => {
                            // Check if this is terminal event
                            let is_ready = matches!(event, LoadingEvent::Ready);

                            // Send event as JSON
                            yield Ok(Event::default().json_data(&event).unwrap());

                            if is_ready {
                                done_state = LoadingState::SendingDone;
                            }
                        }
                        Err(_) => {
                            // Channel closed, send [DONE]
                            done_state = LoadingState::SendingDone;
                        }
                    }
                }
            }
        }
    };

    // Industry standard: 10-second keep-alive (mistral.rs pattern)
    Ok(Sse::new(stream).keep_alive(KeepAlive::new().interval(Duration::from_secs(10))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::sync::broadcast;

    #[test]
    fn test_loading_event_serialization() {
        let event =
            LoadingEvent::LoadingToVram { layers_loaded: 12, layers_total: 32, vram_mb: 2048 };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("\"stage\":\"loading_to_vram\""));
        assert!(json.contains("\"layers_loaded\":12"));
        assert!(json.contains("\"layers_total\":32"));
        assert!(json.contains("\"vram_mb\":2048"));
    }

    #[test]
    fn test_loading_event_ready_serialization() {
        let event = LoadingEvent::Ready;
        let json = serde_json::to_string(&event).unwrap();
        assert_eq!(json, r#"{"stage":"ready"}"#);
    }

    #[tokio::test]
    async fn test_loading_event_channel() {
        let (tx, mut rx) = broadcast::channel::<LoadingEvent>(100);

        // Send loading event
        tx.send(LoadingEvent::LoadingToVram { layers_loaded: 10, layers_total: 32, vram_mb: 1024 })
            .unwrap();

        // Receive event
        let event = rx.recv().await.unwrap();
        assert!(matches!(
            event,
            LoadingEvent::LoadingToVram { layers_loaded: 10, layers_total: 32, vram_mb: 1024 }
        ));
    }

    #[tokio::test]
    async fn test_loading_ready_event() {
        let (tx, mut rx) = broadcast::channel::<LoadingEvent>(100);

        // Send ready event
        tx.send(LoadingEvent::Ready).unwrap();

        // Receive event
        let event = rx.recv().await.unwrap();
        assert!(matches!(event, LoadingEvent::Ready));
    }
}
