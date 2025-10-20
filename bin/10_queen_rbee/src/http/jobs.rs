//! Job submission and streaming endpoints for queen-rbee
//!
//! Created by: TEAM-155
//!
//! Implements the dual-call pattern for job orchestration:
//! 1. POST /jobs - Create job, return job_id + sse_url
//! 2. GET /jobs/{job_id}/stream - Stream job results via SSE
//!
//! Pattern mirrors worker-rbee but for orchestration, not inference.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{sse::Event, Sse},
    Json,
};
use chrono;
use futures::stream::{self, Stream, StreamExt};
use job_registry::{JobRegistry, JobState as RegistryJobState};
use observability_narration_core::Narration;
use queen_rbee_hive_catalog::{HiveCatalog, HiveRecord, HiveStatus};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;
use tokio::process::Command;

// TEAM-155: Actor and action constants for narration
const ACTOR_QUEEN_HTTP: &str = "👑 queen-http";
const ACTION_JOB_CREATE: &str = "job_create";
const ACTION_JOB_STREAM: &str = "job_stream";
const ACTION_ERROR: &str = "error";

/// Job creation request
///
/// TEAM-155: Queen receives job parameters from rbee-keeper
#[derive(Debug, Deserialize)]
pub struct JobRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
}

/// Job creation response
///
/// TEAM-155: Returns job_id and SSE URL for streaming
#[derive(Debug, Serialize)]
pub struct JobResponse {
    pub job_id: String,
    pub sse_url: String,
}

/// Shared state for job endpoints
///
/// TEAM-155: Generic over token type (queen will use String for now)
/// TEAM-156: Added hive catalog for checking available hives
/// Named QueenJobState to avoid conflict with job_registry::JobState
#[derive(Clone)]
pub struct QueenJobState {
    pub registry: Arc<JobRegistry<String>>,
    pub hive_catalog: Arc<HiveCatalog>,
}

/// POST /jobs - Create a new job
///
/// TEAM-155: Mirrors worker-rbee pattern
/// TEAM-156: Added hive catalog checking and "no hives found" narration
/// - Server generates job_id (client doesn't provide it)
/// - Returns JSON with job_id and sse_url
/// - Client then calls GET /jobs/{job_id}/stream for SSE
pub async fn handle_create_job(
    State(state): State<QueenJobState>,
    Json(req): Json<JobRequest>,
) -> Result<Json<JobResponse>, (StatusCode, String)> {
    // TEAM-155: Create job in registry (generates job_id)
    let job_id = state.registry.create_job();

    Narration::new(ACTOR_QUEEN_HTTP, ACTION_JOB_CREATE, &job_id)
        .human(format!("Job {} created for model {}", job_id, req.model))
        .emit();

    // TEAM-156: Check hive catalog for available hives
    let hive_catalog = state.hive_catalog.clone();
    let hives = hive_catalog
        .list_hives()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if hives.is_empty() {
        // TEAM-156: No hives found - send narration via SSE
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        state.registry.set_token_receiver(&job_id, rx);

        Narration::new(ACTOR_QUEEN_HTTP, ACTION_JOB_CREATE, &job_id)
            .human("No hives found in catalog")
            .emit();

        // TEAM-156: Send "no hives found" message to SSE stream
        if tx.send("No hives found.".to_string()).is_err() {
            // Receiver dropped, but that's okay
        }

        // TEAM-157: Add local PC to hive catalog
        let now_ms = chrono::Utc::now().timestamp_millis();
        let localhost_record = HiveRecord {
            id: "localhost".to_string(),
            host: "127.0.0.1".to_string(),
            port: 8600,
            ssh_host: None,
            ssh_port: None,
            ssh_user: None,
            status: HiveStatus::Unknown,
            last_heartbeat_ms: None,
            devices: None, // TEAM-159: Will be populated on first heartbeat
            created_at_ms: now_ms,
            updated_at_ms: now_ms,
        };

        hive_catalog
            .add_hive(localhost_record)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        Narration::new(ACTOR_QUEEN_HTTP, ACTION_JOB_CREATE, &job_id)
            .human("Adding local pc to hive catalog")
            .emit();

        // TEAM-157: Send narration to SSE stream
        if tx.send("Adding local pc to hive catalog.".to_string()).is_err() {
            // Receiver dropped, but that's okay
        }

        // TEAM-157: Start rbee-hive locally on port 8600
        // HARDCODED LOCATION OF RBEE HIVE BINARY FOR NOW!
        let hive_binary = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.join("rbee-hive")))
            .unwrap_or_else(|| std::path::PathBuf::from("./target/debug/rbee-hive"));

        Narration::new(ACTOR_QUEEN_HTTP, ACTION_JOB_CREATE, "localhost")
            .human("Waking up the bee hive at localhost")
            .emit();

        // TEAM-157: Send narration to SSE stream
        if tx.send("Waking up the bee hive at localhost.".to_string()).is_err() {
            // Receiver dropped, but that's okay
        }

        // TEAM-157: Spawn rbee-hive subprocess
        let _hive_process =
            Command::new(&hive_binary).arg("--port").arg("8600").spawn().map_err(|e| {
                Narration::new(ACTOR_QUEEN_HTTP, ACTION_ERROR, "localhost")
                    .human(format!("Failed to start rbee-hive: {}", e))
                    .error_kind("hive_spawn_failed")
                    .emit();
                (StatusCode::INTERNAL_SERVER_ERROR, format!("Failed to start rbee-hive: {}", e))
            })?;

        Narration::new(ACTOR_QUEEN_HTTP, ACTION_JOB_CREATE, "localhost")
            .human("Rbee-hive started, waiting for heartbeat")
            .emit();

        // TEAM-157: Send narration to SSE stream
        if tx.send("Rbee-hive started, waiting for heartbeat.".to_string()).is_err() {
            // Receiver dropped, but that's okay
        }

        // TEAM-157: TODO TEAM-158: Implement heartbeat listener endpoint
        // The hive will automatically send heartbeats to queen
        // For now, just close the stream
        drop(tx);
    }

    // TEAM-155: Return job_id and SSE URL
    let sse_url = format!("/jobs/{}/stream", job_id);

    Ok(Json(JobResponse { job_id, sse_url }))
}

/// GET /jobs/{job_id}/stream - Stream job results via SSE
///
/// TEAM-155: Mirrors worker-rbee pattern
/// - Retrieves job from registry
/// - Takes token receiver (can only be called once!)
/// - Streams events to client
pub async fn handle_stream_job(
    Path(job_id): Path<String>,
    State(state): State<QueenJobState>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, (StatusCode, String)> {
    // TEAM-155: Check if job exists
    if !state.registry.has_job(&job_id) {
        Narration::new(ACTOR_QUEEN_HTTP, ACTION_ERROR, &job_id)
            .human(format!("Job {} not found", job_id))
            .error_kind("job_not_found")
            .emit();

        return Err((StatusCode::NOT_FOUND, format!("Job {} not found", job_id)));
    }

    // TEAM-155: Check job state
    if let Some(RegistryJobState::Failed(error)) = state.registry.get_job_state(&job_id) {
        Narration::new(ACTOR_QUEEN_HTTP, ACTION_ERROR, &job_id)
            .human(format!("Job {} failed: {}", job_id, error))
            .error_kind("job_failed")
            .emit();

        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("Job {} failed: {}", job_id, error),
        ));
    }

    // TEAM-155: Take the receiver from the registry
    // This consumes it - can only be called once per job!
    let mut response_rx = state.registry.take_token_receiver(&job_id).ok_or_else(|| {
        Narration::new(ACTOR_QUEEN_HTTP, ACTION_ERROR, &job_id)
            .human(format!("Job {} has no token receiver", job_id))
            .error_kind("no_token_receiver")
            .emit();

        (StatusCode::INTERNAL_SERVER_ERROR, format!("Job {} has no token receiver", job_id))
    })?;

    Narration::new(ACTOR_QUEEN_HTTP, ACTION_JOB_STREAM, &job_id)
        .human(format!("SSE stream started for job {}", job_id))
        .emit();

    // TEAM-155: Build SSE stream
    // For now, just send a started event and stream tokens
    // Later: Forward from worker SSE stream
    let started_event = format!(
        r#"{{"type":"started","job_id":"{}","started_at":"{}"}}"#,
        job_id,
        chrono::Utc::now().to_rfc3339()
    );

    // TEAM-155: Stream tokens from receiver
    let token_events = Box::pin(async_stream::stream! {
        while let Some(token) = response_rx.recv().await {
            // TEAM-155: For now, just send token as data
            // Later: Parse and forward worker events
            yield Ok(Event::default().data(token));
        }
    });

    let started_stream =
        stream::once(futures::future::ready(Ok(Event::default().data(started_event))));

    let stream_with_done = started_stream
        .chain(token_events)
        .chain(stream::once(futures::future::ready(Ok(Event::default().data("[DONE]")))));

    Ok(Sse::new(stream_with_done))
}
