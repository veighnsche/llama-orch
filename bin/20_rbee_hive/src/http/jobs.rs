//! Job creation and streaming HTTP endpoints
//!
//! TEAM-261: Mirrors queen-rbee pattern for consistency
//!
//! This module is a thin HTTP wrapper that delegates to job_router for business logic.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::sse::{Event, Sse},
    Json,
};
use futures::stream::Stream;
use job_server::JobRegistry;
use observability_narration_core::sse_sink;
use rbee_hive_model_catalog::ModelCatalog; // TEAM-268: Model catalog
use rbee_hive_worker_catalog::WorkerCatalog; // TEAM-274: Worker catalog
use std::{convert::Infallible, sync::Arc};

/// State for HTTP job endpoints
#[derive(Clone)]
pub struct HiveState {
    /// Registry for managing job lifecycle and payloads
    pub registry: Arc<JobRegistry<String>>,
    /// Model catalog for model management
    pub model_catalog: Arc<ModelCatalog>, // TEAM-268: Added
    /// Worker catalog for worker binary management
    pub worker_catalog: Arc<WorkerCatalog>, // TEAM-274: Added
                                            // TODO: Add model_provisioner when TEAM-269 implements it
                                            // TODO: Add worker_registry when implemented
}

/// Convert HTTP state to router state
impl From<HiveState> for crate::job_router::JobState {
    fn from(state: HiveState) -> Self {
        Self {
            registry: state.registry,
            model_catalog: state.model_catalog, // TEAM-268: Added
            worker_catalog: state.worker_catalog, // TEAM-274: Added
        }
    }
}

/// POST /v1/jobs - Create a new job (ALL operations)
///
/// TEAM-261: Thin HTTP wrapper that delegates to job_router::create_job()
///
/// Example payloads (from queen-rbee):
/// - {"operation": "worker_spawn", "hive_id": "localhost", "model": "...", "worker": "cpu", "device": 0}
/// - {"operation": "model_download", "hive_id": "localhost", "model": "..."}
/// - {"operation": "infer", "hive_id": "localhost", "model": "...", "prompt": "...", ...}
pub async fn handle_create_job(
    State(state): State<HiveState>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<crate::job_router::JobResponse>, (StatusCode, String)> {
    // Delegate to router
    crate::job_router::create_job(state.into(), payload)
        .await
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))
}

/// DELETE /v1/jobs/{job_id} - Cancel a job
///
/// TEAM-305-FIX: Allow users to cancel running or queued jobs
///
/// Returns:
/// - 200 OK with job_id if cancelled successfully
/// - 404 NOT FOUND if job doesn't exist or cannot be cancelled
pub async fn handle_cancel_job(
    Path(job_id): Path<String>,
    State(state): State<HiveState>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let cancelled = state.registry.cancel_job(&job_id);
    
    if cancelled {
        Ok(Json(serde_json::json!({
            "job_id": job_id,
            "status": "cancelled"
        })))
    } else {
        Err((
            StatusCode::NOT_FOUND,
            format!("Job {} not found or cannot be cancelled (already completed/failed)", job_id)
        ))
    }
}

/// GET /v1/jobs/{job_id}/stream - Stream job results via SSE
///
/// TEAM-305-FIX: Mirrors queen-rbee pattern
///
/// This handler:
/// 1. Takes the job-specific SSE receiver (MPSC - can only be done once)
/// 2. Triggers job execution (which emits narrations)
/// 3. Streams narration events back to queen-rbee
/// 4. Sends [DONE] marker when complete
/// 5. When receiver drops, sender fails gracefully (natural cleanup)
///
/// Client (queen-rbee) connects here, which triggers job execution and streams results.
pub async fn handle_stream_job(
    Path(job_id): Path<String>,
    State(state): State<HiveState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // Take the receiver (can only be done once per job)
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
                // MPSC recv() returns Option<T>
                event_opt = sse_rx.recv() => {
                    match event_opt {
                        Some(event) => {
                            received_first_event = true;
                            last_event_time = std::time::Instant::now();
                            // Use pre-formatted text from narration-core
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

        // Cleanup - remove sender from HashMap to prevent memory leak
        // Receiver is already dropped by moving out of this scope
        sse_sink::remove_job_channel(&job_id_for_stream);
    };

    Sse::new(combined_stream)
}
