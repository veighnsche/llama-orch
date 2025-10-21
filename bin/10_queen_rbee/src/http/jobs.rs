//! Job creation and streaming HTTP endpoints
//!
//! TEAM-186: Job-based architecture - ALL operations go through POST /v1/jobs
//! TEAM-189: Fixed SSE streaming - subscribe to narration broadcaster
//!
//! This module is a thin HTTP wrapper that delegates to job_router for business logic.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::sse::{Event, Sse},
    Json,
};
use futures::stream::{Stream, StreamExt};
use job_registry::JobRegistry;
use observability_narration_core::sse_sink;
use rbee_config::RbeeConfig;
use std::{convert::Infallible, sync::Arc};

/// TEAM-204: Drop guard ensures job channel cleanup even on panic/early return
struct JobChannelGuard {
    job_id: String,
}

impl Drop for JobChannelGuard {
    fn drop(&mut self) {
        sse_sink::remove_job_channel(&self.job_id);
    }
}

/// State for HTTP job endpoints
#[derive(Clone)]
pub struct SchedulerState {
    /// Registry for managing job lifecycle and payloads
    pub registry: Arc<JobRegistry<String>>,
    /// TEAM-194: File-based config (replaces hive_catalog)
    pub config: Arc<RbeeConfig>,
    /// TEAM-190: Runtime registry for live hive/worker state
    pub hive_registry: Arc<queen_rbee_hive_registry::HiveRegistry>,
}

/// Convert HTTP state to router state
impl From<SchedulerState> for crate::job_router::JobState {
    fn from(state: SchedulerState) -> Self {
        Self {
            registry: state.registry,
            config: state.config,               // TEAM-194
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
/// TEAM-204: Drop guard ensures cleanup on panic/early return
///
/// This handler:
/// 1. Subscribes to the job-specific SSE narration broadcaster
/// 2. Triggers job execution (which emits narrations)
/// 3. Streams narration events to the client
/// 4. Also streams token results (for inference operations)
/// 5. Sends [DONE] marker when complete
/// 6. Cleans up job channel to prevent memory leaks (guaranteed by drop guard)
///
/// Client connects here, which triggers job execution and streams results.
pub async fn handle_stream_job(
    Path(job_id): Path<String>,
    State(state): State<SchedulerState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // TEAM-204: Drop guard ensures cleanup even if we panic or return early
    let _guard = JobChannelGuard { job_id: job_id.clone() };
    
    // TEAM-204: Check if job channel exists before proceeding
    let sse_rx_opt = sse_sink::subscribe_to_job(&job_id);
    
    // Trigger job execution (spawns in background) - do this even if channel missing
    let _token_stream = crate::job_router::execute_job(job_id.clone(), state.into()).await;

    // Single stream that handles both error and success cases
    let combined_stream = async_stream::stream! {
        // TEAM-204: Handle missing channel gracefully
        let Some(mut sse_rx) = sse_rx_opt else {
            yield Ok(Event::default().data("ERROR: Job channel not found. This may indicate a race condition or job creation failure."));
            return;
        };

        // Give the background task a moment to start executing
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

        let mut last_event_time = std::time::Instant::now();
        let completion_timeout = std::time::Duration::from_millis(2000);
        let mut received_first_event = false;

        loop {
            // Wait for either a narration event or timeout
            let timeout_fut = tokio::time::sleep(completion_timeout);
            tokio::pin!(timeout_fut);

            tokio::select! {
                // Receive narration events from SSE broadcaster
                result = sse_rx.recv() => {
                    match result {
                        Ok(event) => {
                            received_first_event = true;
                            last_event_time = std::time::Instant::now();
                            // TEAM-201: Use pre-formatted text from narration-core (no manual formatting!)
                            yield Ok(Event::default().data(&event.formatted));
                        }
                        Err(_) => {
                            // Broadcaster closed (shouldn't happen)
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
                        // TEAM-204: Cleanup happens automatically via drop guard
                        break;
                    }
                }
            }
        }
    };

    Sse::new(combined_stream)
}
