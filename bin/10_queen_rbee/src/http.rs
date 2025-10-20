//! HTTP endpoints for queen-rbee
//!
//! TEAM-164: All HTTP-specific code lives HERE
//!
//! ⚠️  WHY THIS FILE EXISTS:
//!
//! Moving HTTP endpoint handlers to crates requires those crates to depend on:
//! - axum (HTTP framework)
//! - serde (JSON serialization)
//! - HTTP-specific types (StatusCode, Json, State, etc.)
//!
//! This pollutes pure business logic crates with HTTP dependencies.
//!
//! ARCHITECTURE DECISION:
//! - Business logic → Lives in crates (pure Rust, no HTTP)
//! - HTTP wrappers → Live HERE (thin wrappers that call crate functions)
//!
//! This keeps crates clean and reusable in non-HTTP contexts.

pub mod narration_stream;

use async_trait::async_trait;
use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::sse::{Event, Sse},
    Json,
};
use futures::stream::{self, Stream};
use job_registry::JobRegistry;
use observability_narration_core::Narration;
use queen_rbee_hive_catalog::HiveCatalog;
use queen_rbee_hive_lifecycle;
use queen_rbee_scheduler;
use rbee_heartbeat::traits::{DetectionError, DeviceDetector, DeviceResponse};
use rbee_heartbeat::HiveHeartbeatPayload;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::Arc;

const ACTOR_QUEEN_HTTP: &str = "👑 queen-http";
const ACTION_JOB_CREATE: &str = "job_create";
const ACTION_JOB_STREAM: &str = "job_stream";

// ============================================================================
// HIVE START ENDPOINT
// ============================================================================
// WHY THIS CAN'T LIVE IN hive-lifecycle CRATE:
//
// This function uses:
// - axum::extract::State     ← HTTP framework dependency
// - axum::http::StatusCode   ← HTTP framework dependency
// - axum::Json               ← HTTP framework dependency
// - Result<(StatusCode, Json<T>), (StatusCode, String)> ← HTTP-specific return type
//
// If we put this in hive-lifecycle crate, we'd need to:
// 1. Add axum as dependency
// 2. Add serde as dependency
// 3. Add #[cfg(feature = "http")] everywhere
// 4. Pollute a pure business logic crate with HTTP concerns
//
// SOLUTION: Keep HTTP wrapper here, call pure business logic from crate
// ============================================================================

#[derive(Debug, Serialize)]
pub struct HiveStartResponse {
    pub hive_url: String,
    pub hive_id: String,
    pub port: u16,
}

pub type HiveStartState = Arc<HiveCatalog>;

/// POST /hive/start - Start a hive
///
/// TEAM-164: Thin HTTP wrapper around hive-lifecycle::execute_hive_start()
/// Follows Command Pattern (see CRATE_INTERFACE_STANDARD.md)
pub async fn handle_hive_start(
    State(catalog): State<HiveStartState>,
) -> Result<(StatusCode, Json<HiveStartResponse>), (StatusCode, String)> {
    // Create domain request
    let request = queen_rbee_hive_lifecycle::HiveStartRequest {
        queen_url: "http://localhost:8500".to_string(),
    };

    // Call pure business logic from crate (no HTTP dependencies)
    let response = queen_rbee_hive_lifecycle::execute_hive_start(Arc::clone(&catalog), request)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Convert domain response to HTTP response
    Ok((
        StatusCode::OK,
        Json(HiveStartResponse {
            hive_url: response.hive_url,
            hive_id: response.hive_id,
            port: response.port,
        }),
    ))
}

// ============================================================================
// JOB CREATE ENDPOINT
// ============================================================================
// WHY THIS CAN'T LIVE IN scheduler CRATE:
//
// This function uses:
// - axum::extract::State     ← HTTP framework dependency
// - axum::http::StatusCode   ← HTTP framework dependency
// - axum::Json               ← HTTP framework dependency
// - #[derive(Deserialize)]   ← serde dependency for HTTP request parsing
// - #[derive(Serialize)]     ← serde dependency for HTTP response
//
// The scheduler crate has pure business logic types (JobRequest, JobResponse)
// that don't need serde. Adding HTTP here would force serde on the crate.
//
// SOLUTION: HTTP types here, call pure business logic from crate
// ============================================================================

#[derive(Debug, Deserialize)]
pub struct HttpJobRequest {
    pub model: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub temperature: f32,
}

#[derive(Debug, Serialize)]
pub struct HttpJobResponse {
    pub job_id: String,
    pub sse_url: String,
}

#[derive(Clone)]
pub struct SchedulerState {
    pub registry: Arc<JobRegistry<String>>,
    pub hive_catalog: Arc<HiveCatalog>,
}

/// POST /jobs - Create a new job
///
/// TEAM-164: Thin HTTP wrapper around scheduler::orchestrate_job()
/// Follows Command Pattern (see CRATE_INTERFACE_STANDARD.md)
pub async fn handle_create_job(
    State(state): State<SchedulerState>,
    Json(req): Json<HttpJobRequest>,
) -> Result<Json<HttpJobResponse>, (StatusCode, String)> {
    // Convert HTTP request to domain request
    let request = queen_rbee_scheduler::JobRequest {
        model: req.model,
        prompt: req.prompt,
        max_tokens: req.max_tokens,
        temperature: req.temperature,
    };

    // Call pure business logic from crate (no HTTP dependencies)
    let response =
        queen_rbee_scheduler::orchestrate_job(state.registry, state.hive_catalog, request)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Convert domain response to HTTP response
    Ok(Json(HttpJobResponse { job_id: response.job_id, sse_url: response.sse_url }))
}

// ============================================================================
// JOB STREAM ENDPOINT
// ============================================================================
// WHY THIS STAYS HERE:
//
// This function uses:
// - axum::extract::Path      ← HTTP framework dependency
// - axum::extract::State     ← HTTP framework dependency
// - axum::response::sse::Sse ← HTTP framework dependency (Server-Sent Events)
// - futures::stream::Stream  ← Async stream for SSE
//
// SSE is inherently HTTP-specific. There's no "pure business logic" version.
// This is genuinely HTTP-only code.
// ============================================================================

/// GET /jobs/{job_id}/stream - Stream job results via SSE
pub async fn handle_stream_job(
    Path(job_id): Path<String>,
    State(registry): State<Arc<JobRegistry<String>>>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    Narration::new(ACTOR_QUEEN_HTTP, ACTION_JOB_STREAM, &job_id)
        .human(format!("Streaming job {}", job_id))
        .emit();

    let receiver = registry.take_token_receiver(&job_id);

    let stream = stream::unfold(receiver, |mut rx_opt| async move {
        match rx_opt {
            Some(mut rx) => match rx.recv().await {
                Some(token) => {
                    let event = Event::default().data(token);
                    Some((Ok(event), Some(rx)))
                }
                None => None,
            },
            None => None,
        }
    });

    Sse::new(stream)
}

// ============================================================================
// HEARTBEAT ENDPOINT
// ============================================================================
// WHY THIS STAYS HERE:
//
// This function uses:
// - axum::extract::State     ← HTTP framework dependency
// - axum::http::StatusCode   ← HTTP framework dependency
// - axum::Json               ← HTTP framework dependency
//
// The rbee-heartbeat crate has pure business logic (handle_hive_heartbeat)
// that works with trait objects. Adding HTTP here would pollute that crate.
//
// SOLUTION: HTTP wrapper here, call pure business logic from crate
// ============================================================================

#[derive(Clone)]
pub struct HeartbeatState {
    pub hive_catalog: Arc<HiveCatalog>,
    pub device_detector: Arc<HttpDeviceDetector>,
}

#[derive(Serialize)]
pub struct HttpHeartbeatAcknowledgement {
    pub status: String,
    pub message: String,
}

/// POST /heartbeat - Handle hive heartbeat
pub async fn handle_heartbeat(
    State(state): State<HeartbeatState>,
    Json(payload): Json<HiveHeartbeatPayload>,
) -> Result<Json<HttpHeartbeatAcknowledgement>, (StatusCode, String)> {
    // Call binary-specific heartbeat logic
    let response =
        crate::heartbeat::handle_hive_heartbeat(state.hive_catalog, payload, state.device_detector)
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(HttpHeartbeatAcknowledgement { status: response.status, message: response.message }))
}

// ============================================================================
// HTTP DEVICE DETECTOR
// ============================================================================
// WHY THIS CAN'T LIVE IN hive-lifecycle OR rbee-heartbeat CRATE:
//
// This struct uses:
// - reqwest::Client          ← HTTP client library dependency
// - async_trait::async_trait ← Async trait dependency
//
// The DeviceDetector trait is defined in rbee-heartbeat and is pure/abstract.
// HttpDeviceDetector is ONE IMPLEMENTATION that uses HTTP.
//
// Other implementations could use:
// - SSH commands
// - Local system calls
// - Mock/test implementations
//
// If we put HttpDeviceDetector in rbee-heartbeat, we'd force reqwest on
// everyone, even those using SSH or local implementations.
//
// If we put it in hive-lifecycle, we'd force reqwest on that crate even
// though the core lifecycle logic doesn't need HTTP.
//
// SOLUTION: HTTP-specific implementation lives here in HTTP layer
// ============================================================================

/// HTTP-based device detector
///
/// Makes HTTP GET requests to hive's /v1/devices endpoint
pub struct HttpDeviceDetector {
    client: reqwest::Client,
}

impl HttpDeviceDetector {
    pub fn new() -> Self {
        Self { client: reqwest::Client::new() }
    }
}

impl Default for HttpDeviceDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DeviceDetector for HttpDeviceDetector {
    async fn detect_devices(&self, hive_url: &str) -> Result<DeviceResponse, DetectionError> {
        let url = format!("{}/v1/devices", hive_url);

        self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| DetectionError::Http(e.to_string()))?
            .json()
            .await
            .map_err(|e| DetectionError::Parse(e.to_string()))
    }
}
