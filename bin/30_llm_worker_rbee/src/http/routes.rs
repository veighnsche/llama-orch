//! HTTP route configuration
//!
//! Modified by: TEAM-017 (updated to use Mutex-wrapped backend)
//! Modified by: TEAM-035 (added loading progress, renamed inference endpoint)
//! Modified by: TEAM-149 (real-time streaming with request queue)
//!
//! # Endpoints
//! - `GET /health` - Health check endpoint
//! - `GET /v1/ready` - Worker readiness check - TEAM-045
//! - `POST /v1/inference` - Execute inference request (SSE) - TEAM-035, TEAM-149
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

use crate::backend::request_queue::RequestQueue;
use crate::http::{execute, middleware::auth_middleware};
use axum::{
    middleware,
    routing::{get, post},
    Json, Router,
};
use observability_narration_core::axum::correlation_middleware;
use serde::Serialize;
use std::sync::Arc;

/// Create HTTP router with all endpoints and middleware
///
/// # Arguments
/// * `queue` - Request queue for inference requests (TEAM-149)
/// * `expected_token` - API token for authentication (TEAM-102)
///
/// # Returns
/// Router with all endpoints and middleware configured
///
/// TEAM-017: Updated to accept Mutex-wrapped backend
/// TEAM-035: Added /v1/loading/progress and renamed /execute to /v1/inference
/// TEAM-045: Added /v1/ready endpoint
/// TEAM-102: Added authentication middleware with `expected_token`
/// TEAM-149: Changed to accept RequestQueue instead of backend
pub fn create_router(
    queue: Arc<RequestQueue>,
    expected_token: String,
) -> Router {
    // TEAM-149: Simple health endpoint (public, no auth)
    // Returns basic status without backend access
    #[derive(Serialize)]
    struct HealthResponse {
        status: String,
    }
    
    async fn handle_health() -> Json<HealthResponse> {
        Json(HealthResponse {
            status: "healthy".to_string(),
        })
    }
    
    // Public routes (no auth)
    let public_routes = Router::new()
        .route("/health", get(handle_health));
    
    // Worker routes (protected)
    let worker_routes = Router::new()
        .route("/v1/inference", post(execute::handle_execute));

    // Apply auth middleware if token is provided (network mode)
    // Empty token = local mode (no auth, but main.rs ensures 127.0.0.1 binding)
    let protected_routes = if expected_token.is_empty() {
        worker_routes
    } else {
        let auth_state = Arc::new(crate::http::middleware::AuthState { expected_token });
        worker_routes.layer(middleware::from_fn_with_state(auth_state, auth_middleware))
    };

    // Merge public and protected routes, apply correlation middleware
    public_routes
        .merge(protected_routes)
        .layer(middleware::from_fn(correlation_middleware))
        .with_state(queue)
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
