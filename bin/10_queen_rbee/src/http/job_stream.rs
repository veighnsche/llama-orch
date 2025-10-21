//! Job execution SSE streaming endpoint
//!
//! TEAM-186: Client connects here to trigger job execution and stream results

use axum::{
    extract::{Path, State},
    response::sse::{Event, Sse},
};
use futures::stream::{Stream, StreamExt};
use observability_narration_core::Narration;
use std::convert::Infallible;

use super::jobs::SchedulerState;

const ACTOR_QUEEN_HTTP: &str = "👑 queen-http";
const ACTION_JOB_STREAM: &str = "job_stream";

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
