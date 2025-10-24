//! Job creation and streaming HTTP endpoints
//!
//! TEAM-186: Job-based architecture - ALL operations go through POST /v1/jobs
//! TEAM-189: Fixed SSE streaming - subscribe to narration broadcaster
//! TEAM-217: Investigated Oct 22, 2025 - Behavior inventory complete
//!
//! This module is a thin HTTP wrapper that delegates to job_router for business logic.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::sse::{Event, Sse},
    Json,
};
use futures::stream::{Stream, StreamExt};
use job_server::JobRegistry;
use observability_narration_core::sse_sink;
// TEAM-290: DELETED rbee_config import (file-based config deprecated)
use std::{convert::Infallible, sync::Arc};

// TEAM-205: Removed JobChannelGuard - it was dropping too early!
// The receiver being taken and eventually dropped provides natural cleanup.
// When the receiver drops, the sender's try_send will fail gracefully.

/// State for HTTP job endpoints
#[derive(Clone)]
pub struct SchedulerState {
    /// Registry for managing job lifecycle and payloads
    pub registry: Arc<JobRegistry<String>>,
    // TEAM-290: DELETED config field (file-based config deprecated)
    /// TEAM-190: Runtime registry for live hive/worker state
    /// TEAM-262: Renamed to WorkerRegistry (field name kept for compatibility)
    pub hive_registry: Arc<queen_rbee_worker_registry::WorkerRegistry>,
}

/// Convert HTTP state to router state
impl From<SchedulerState> for crate::job_router::JobState {
    fn from(state: SchedulerState) -> Self {
        Self {
            registry: state.registry,
            // TEAM-290: DELETED config field (file-based config deprecated)
            hive_registry: state.hive_registry, // TEAM-190
        }
    }
}

/// POST /v1/jobs - Create a new job (ALL operations)
///
/// Thin HTTP wrapper that delegates to job_router::create_job().
///
/// Example payloads:
/// - {"operation": "hive_list"}
/// - {"operation": "worker_spawn", "hive_id": "localhost", "model": "...", "worker": "cpu", "device": 0}
/// - {"operation": "infer", "hive_id": "localhost", "model": "...", "prompt": "...", ...}
pub async fn handle_create_job(
    State(state): State<SchedulerState>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<crate::job_router::JobResponse>, (StatusCode, String)> {
    // Delegate to router
    crate::job_router::create_job(state.into(), payload)
        .await
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

/// GET /v1/jobs/{job_id}/stream - Stream job results via SSE
///
/// TEAM-189: Fixed to subscribe to SSE narration broadcaster
/// TEAM-200: Subscribe to JOB-SPECIFIC channel (not global!)
/// TEAM-204: Proper error handling instead of panic
/// TEAM-205: SIMPLIFIED - Use MPSC receiver (no broadcast complexity)
/// TEAM-205: Natural cleanup - receiver drop triggers sender cleanup
///
/// This handler:
/// 1. Takes the job-specific SSE receiver (MPSC - can only be done once)
/// 2. Triggers job execution (which emits narrations)
/// 3. Streams narration events to the client
/// 4. Also streams token results (for inference operations)
/// 5. Sends [DONE] marker when complete
/// 6. When receiver drops, sender fails gracefully (natural cleanup)
///
/// Client connects here, which triggers job execution and streams results.
pub async fn handle_stream_job(
    Path(job_id): Path<String>,
    State(state): State<SchedulerState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // TEAM-205: Take the receiver (can only be done once per job)
    let sse_rx_opt = sse_sink::take_job_receiver(&job_id);

    // Trigger job execution (spawns in background) - do this even if channel missing
    let _token_stream = crate::job_router::execute_job(job_id.clone(), state.into()).await;

    // Single stream that handles both error and success cases
    let job_id_for_stream = job_id.clone();
    let combined_stream = async_stream::stream! {
        // Check if channel exists
        let Some(mut sse_rx) = sse_rx_opt else {
            yield Ok(Event::default().data("ERROR: Job channel not found. This may indicate a race condition or job creation failure."));
            return;
        };

        let mut last_event_time = std::time::Instant::now();
        let completion_timeout = std::time::Duration::from_millis(2000);
        let mut received_first_event = false;

        loop {
            // Wait for either a narration event or timeout
            let timeout_fut = tokio::time::sleep(completion_timeout);
            tokio::pin!(timeout_fut);

            tokio::select! {
                // TEAM-205: MPSC recv() returns Option<T> (simpler than broadcast's Result)
                event_opt = sse_rx.recv() => {
                    match event_opt {
                        Some(event) => {
                            received_first_event = true;
                            last_event_time = std::time::Instant::now();
                            // TEAM-201: Use pre-formatted text from narration-core
                            yield Ok(Event::default().data(&event.formatted));
                        }
                        None => {
                            // Sender dropped (job completed)
                            if received_first_event {
                                yield Ok(Event::default().data("[DONE]"));
                            }
                            break;
                        }
                    }
                }
                // Timeout: if no events for 2 seconds after first event, we're done
                _ = &mut timeout_fut, if received_first_event => {
                    if last_event_time.elapsed() >= completion_timeout {
                        yield Ok(Event::default().data("[DONE]"));
                        break;
                    }
                }
            }
        }

        // TEAM-205: Cleanup - remove sender from HashMap to prevent memory leak
        // Receiver is already dropped by moving out of this scope
        sse_sink::remove_job_channel(&job_id_for_stream);
    };

    Sse::new(combined_stream)
}
