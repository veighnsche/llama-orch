//! Job creation and streaming HTTP endpoints
//!
//! TEAM-186: Job-based architecture - ALL operations go through POST /v1/jobs
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
use queen_rbee_hive_catalog::HiveCatalog;
use std::{convert::Infallible, sync::Arc};

/// State for HTTP job endpoints
#[derive(Clone)]
pub struct SchedulerState {
    /// Registry for managing job lifecycle and payloads
    pub registry: Arc<JobRegistry<String>>,
    /// Catalog of registered hives for job routing
    pub hive_catalog: Arc<HiveCatalog>,
}

/// Convert HTTP state to router state
impl From<SchedulerState> for crate::job_router::JobState {
    fn from(state: SchedulerState) -> Self {
        Self { registry: state.registry, hive_catalog: state.hive_catalog }
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
/// Thin HTTP wrapper that delegates to job_router::execute_job().
///
/// Client connects here, which triggers job execution and streams results.
pub async fn handle_stream_job(
    Path(job_id): Path<String>,
    State(state): State<SchedulerState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    // Delegate to router for execution
    let token_stream = crate::job_router::execute_job(job_id, state.into()).await;

    // Convert String stream to SSE Event stream
    let event_stream = token_stream.map(|data| Ok(Event::default().data(data)));

    Sse::new(event_stream)
}
