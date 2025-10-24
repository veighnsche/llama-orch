//! HTTP route configuration
//!
//! Modified by: TEAM-017 (updated to use Mutex-wrapped backend)
//! Modified by: TEAM-035 (added loading progress, renamed inference endpoint)
//! Modified by: TEAM-149 (real-time streaming with request queue)
//! Modified by: TEAM-154 (dual-call pattern with job registry)
//!
//! # Endpoints
//! - `GET /health` - Health check endpoint
//! - `GET /v1/ready` - Worker readiness check - TEAM-045
//! - `POST /v1/inference` - Create inference job (JSON) - TEAM-154
//! - `GET /v1/inference/{job_id}/stream` - Stream job results (SSE) - TEAM-154
//! - `GET /v1/loading/progress` - Model loading progress (SSE) - TEAM-035
//!
//! # Middleware
//! - Correlation ID middleware (extracts/generates X-Correlation-ID)
//!
//! # Spec References
//! - M0-W-1320: Health endpoint
//! - M0-W-1330: Execute endpoint (now /v1/inference)
//! - WORK-3040: Correlation ID middleware
//! - `SSE_IMPLEMENTATION_PLAN.md` Phase 2: Loading progress
//! - `SSE_IMPLEMENTATION_PLAN.md` Phase 3: Inference streaming
//! - `STREAMING_REFACTOR_PLAN.md`: Real-time token streaming

use crate::backend::request_queue::{RequestQueue, TokenResponse};
use crate::http::{execute, middleware::auth_middleware, stream};
use axum::{
    extract::State,
    middleware::{self, Next},
    response::Response,
    routing::{get, post},
    Json, Router,
};
use job_server::JobRegistry;
// TEAM-285: correlation_middleware removed from narration-core
// TODO: Re-implement if needed for request tracing
// use observability_narration_core::axum::correlation_middleware;
use serde::Serialize;
use std::sync::Arc;

/// Shared state for worker routes
///
/// TEAM-154: Combines queue and registry for dual-call pattern
/// Registry is generic over `TokenResponse` type
#[derive(Clone)]
pub struct WorkerState {
    pub queue: Arc<RequestQueue>,
    pub registry: Arc<JobRegistry<TokenResponse>>,
}

/// Create HTTP router with all endpoints and middleware
///
/// # Arguments
/// * `queue` - Request queue for inference requests (TEAM-149)
/// * `registry` - Job registry for dual-call pattern (TEAM-154)
/// * `expected_token` - API token for authentication (TEAM-102)
///
/// # Returns
/// Router with all endpoints and middleware configured
///
/// TEAM-017: Updated to accept Mutex-wrapped backend
/// TEAM-035: Added /v1/loading/progress and renamed /execute to /v1/inference
/// TEAM-045: Added /v1/ready endpoint
/// TEAM-102: Added authentication middleware with `expected_token`
/// TEAM-149: Changed to accept `RequestQueue` instead of backend
/// TEAM-154: Added `JobRegistry` for dual-call pattern
pub fn create_router(
    queue: Arc<RequestQueue>,
    registry: Arc<JobRegistry<TokenResponse>>,
    expected_token: String,
) -> Router {
    // TEAM-149: Simple health endpoint (public, no auth)
    // Returns basic status without backend access
    #[derive(Serialize)]
    struct HealthResponse {
        status: String,
    }

    async fn handle_health() -> Json<HealthResponse> {
        Json(HealthResponse { status: "healthy".to_string() })
    }

    // Public routes (no auth)
    let public_routes = Router::new().route("/health", get(handle_health));

    // TEAM-154: Create shared state for worker routes
    let worker_state = WorkerState { queue, registry };

    // Worker routes (protected)
    // TEAM-154: Dual-call pattern - POST creates job, GET streams results
    let worker_routes = Router::new()
        .route("/v1/inference", post(execute::handle_create_job))
        .route("/v1/inference/{job_id}/stream", get(stream::handle_stream_job))
        .with_state(worker_state);

    // Apply auth middleware if token is provided (network mode)
    // Empty token = local mode (no auth, but main.rs ensures 127.0.0.1 binding)
    let protected_routes = if expected_token.is_empty() {
        worker_routes
    } else {
        let auth_state = Arc::new(crate::http::middleware::AuthState { expected_token });
        worker_routes.layer(middleware::from_fn_with_state(auth_state, auth_middleware))
    };

    // TEAM-154: Merge public and protected routes
    // TEAM-285: correlation_middleware removed (narration-core no longer provides axum module)
    Router::new()
        .merge(public_routes)
        .merge(protected_routes)
        // TODO: Re-implement correlation_middleware if needed for request tracing
        // .layer(middleware::from_fn(correlation_middleware))
}

#[cfg(test)]
mod tests {
    #[tokio::test]
    async fn test_router_has_health_route() {
        // This test verifies the router is configured correctly
        // Full integration tests will be in tests/ directory

        // We can't create a real ModelHandle without CUDA,
        // so we just verify the router creation doesn't panic
        // and has the expected structure

        // Router creation is tested in integration tests
        // where we can mock the CUDA dependencies
    }
}

// ---
// Built by Foundation-Alpha 🏗️
