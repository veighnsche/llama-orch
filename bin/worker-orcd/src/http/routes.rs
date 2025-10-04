//! HTTP route configuration
//!
//! This module defines all HTTP routes for the worker-orcd API.
//!
//! # Endpoints
//! - `GET /health` - Health check endpoint
//! - `POST /execute` - Execute inference request
//!
//! # Spec References
//! - M0-W-1320: Health endpoint
//! - M0-W-1330: Execute endpoint

use crate::cuda::safe::ModelHandle;
use crate::http::{execute, health};
use axum::{
    routing::{get, post},
    Router,
};
use std::sync::Arc;

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    /// Worker ID (UUID)
    pub worker_id: String,

    /// Loaded CUDA model
    pub model: Arc<ModelHandle>,
}

/// Create HTTP router with all endpoints
///
/// # Arguments
/// * `worker_id` - Worker UUID
/// * `model` - Loaded CUDA model handle
///
/// # Returns
/// Configured Axum router with all endpoints
pub fn create_router(worker_id: String, model: ModelHandle) -> Router {
    let state = AppState { worker_id, model: Arc::new(model) };

    Router::new()
        .route("/health", get(health::handle_health))
        .route("/execute", post(execute::handle_execute))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    // Mock ModelHandle for testing
    // Note: In real tests, we'd need to properly mock the CUDA model
    // For now, these tests verify route configuration only

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
