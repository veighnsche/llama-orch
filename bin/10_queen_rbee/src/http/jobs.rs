//! Job creation and streaming HTTP endpoints
//!
//! TEAM-186: Job-based architecture - ALL operations go through POST /v1/jobs

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::sse::{Event, Sse},
    Json,
};
use futures::stream::{Stream, StreamExt};
use job_registry::JobRegistry;
use observability_narration_core::Narration;
use queen_rbee_hive_catalog::HiveCatalog;
use serde::Serialize;
use std::{convert::Infallible, sync::Arc};

const ACTOR_QUEEN_HTTP: &str = "👑 queen-http";
const ACTION_JOB_CREATE: &str = "job_create";
const ACTION_JOB_STREAM: &str = "job_stream";

/// HTTP response for job creation
#[derive(Debug, Serialize)]
pub struct HttpJobResponse {
    /// Unique identifier for the created job
    pub job_id: String,
    /// SSE endpoint URL where the client can stream job results
    pub sse_url: String,
}

/// State for the job scheduler endpoint
#[derive(Clone)]
pub struct SchedulerState {
    /// Registry for managing job lifecycle and payloads
    pub registry: Arc<JobRegistry<String>>,
    /// Catalog of registered hives for job routing
    pub hive_catalog: Arc<HiveCatalog>,
}

/// POST /v1/jobs - Create a new job (ALL operations)
///
/// TEAM-186: Generic job endpoint that routes based on "operation" field
/// 
/// **Job Lifecycle:**
/// 1. POST /v1/jobs - Creates job, stores payload, returns job_id + sse_url
/// 2. GET /v1/jobs/{job_id}/stream - Client connects, job starts executing
///
/// This allows the client to connect to the SSE stream before execution starts,
/// ensuring no events are missed.
///
/// Example payloads:
/// - {"operation": "hive_list"}
/// - {"operation": "worker_spawn", "hive_id": "localhost", "model": "...", "worker": "cpu", "device": 0}
/// - {"operation": "infer", "hive_id": "localhost", "model": "...", "prompt": "...", ...}
pub async fn handle_create_job(
    State(state): State<SchedulerState>,
    Json(payload): Json<serde_json::Value>,
) -> Result<Json<HttpJobResponse>, (StatusCode, String)> {
    // TEAM-186: Create job in registry but DON'T execute yet
    let job_id = state.registry.create_job();
    let sse_url = format!("/v1/jobs/{}/stream", job_id);

    // TEAM-186: Store payload in job registry for later execution
    // Job will execute when client connects to SSE stream
    state.registry.set_payload(&job_id, payload);

    Narration::new(ACTOR_QUEEN_HTTP, ACTION_JOB_CREATE, &job_id)
        .human(format!("Job {} created, waiting for client connection", job_id))
        .emit();

    Ok(Json(HttpJobResponse { job_id, sse_url }))
}

/// GET /v1/jobs/{job_id}/stream - Stream job results via SSE
///
/// TEAM-186: Client connects here, which triggers job execution
/// 
/// **Flow:**
/// 1. Client connects to SSE stream
/// 2. Retrieve pending job payload
/// 3. Start job execution in background
/// 4. Stream results as they arrive
pub async fn handle_stream_job(
    Path(job_id): Path<String>,
    State(state): State<SchedulerState>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    Narration::new(ACTOR_QUEEN_HTTP, ACTION_JOB_STREAM, &job_id)
        .human(format!("Client connected, starting job {}", job_id))
        .emit();

    // TEAM-186: Use shared execute_and_stream helper from job-registry
    let registry = state.registry.clone();
    let hive_catalog = state.hive_catalog.clone();
    
    let token_stream = job_registry::execute_and_stream(
        job_id,
        registry.clone(),
        move |_job_id, payload| {
            let router_state = crate::job_router::JobRouterState {
                registry,
                hive_catalog,
            };
            async move {
                crate::job_router::route_job(router_state, payload)
                    .await
                    .map(|_| ())  // Discard JobResponse, we only care about success/failure
                    .map_err(|e| anyhow::anyhow!(e))
            }
        },
    ).await;

    // Convert String stream to SSE Event stream
    let event_stream = token_stream.map(|data| Ok(Event::default().data(data)));
    
    Sse::new(event_stream)
}
