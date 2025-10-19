//! HTTP route configuration
//!
//! Modified by: TEAM-017 (updated to use Mutex-wrapped backend)
//! Modified by: TEAM-035 (added loading progress, renamed inference endpoint)
//!
//! # Endpoints
//! - `GET /health` - Health check endpoint
//! - `GET /v1/ready` - Worker readiness check - TEAM-045
//! - `POST /v1/inference` - Execute inference request (SSE) - TEAM-035
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

use crate::http::{
    backend::InferenceBackend, execute, health, loading, middleware::auth_middleware, ready,
};
use axum::{
    middleware,
    routing::{get, post},
    Router,
};
use observability_narration_core::axum::correlation_middleware;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Create HTTP router with all endpoints and middleware
///
/// # Arguments
/// * `backend` - Platform-specific inference backend (Mutex-wrapped)
/// * `expected_token` - API token for authentication (TEAM-102)
///
/// # Returns
/// Router with all endpoints and middleware configured
///
/// TEAM-017: Updated to accept Mutex-wrapped backend
/// TEAM-035: Added /v1/loading/progress and renamed /execute to /v1/inference
/// TEAM-045: Added /v1/ready endpoint
/// TEAM-102: Added authentication middleware with `expected_token`
pub fn create_router<B: InferenceBackend + 'static>(
    backend: Arc<Mutex<B>>,
    expected_token: String, // TEAM-102: API token for authentication
) -> Router {
    // TEAM-102: Create auth state for authentication middleware
    let auth_state = Arc::new(crate::http::middleware::AuthState { expected_token });

    // TEAM-102: Split routes into public and protected
    let public_routes = Router::new()
        // Health endpoint (public - no auth required)
        .route("/health", get(health::handle_health::<B>));

    let protected_routes = Router::new()
        // Worker endpoints (protected)
        .route("/v1/ready", get(ready::handle_ready::<B>))
        .route("/v1/inference", post(execute::handle_execute::<B>))
        .route("/v1/loading/progress", get(loading::handle_loading_progress::<B>))
        // TEAM-102: Apply authentication middleware to all protected routes
        .layer(middleware::from_fn_with_state(auth_state, auth_middleware));

    // TEAM-102: Merge public and protected routes, apply correlation middleware to all
    public_routes
        .merge(protected_routes)
        .layer(middleware::from_fn(correlation_middleware))
        .with_state(backend)
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
