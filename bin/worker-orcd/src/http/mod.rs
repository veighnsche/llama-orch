//! HTTP API for worker-orcd

mod execute;
mod health;

use crate::cuda::safe::ModelHandle;
use axum::{routing::{get, post}, Router};
use std::sync::Arc;

/// Shared application state
pub struct AppState {
    pub worker_id: String,
    pub model: Arc<ModelHandle>,
}

/// Create HTTP router with all endpoints
pub fn create_router(worker_id: String, model: ModelHandle) -> Router {
    let state = Arc::new(AppState {
        worker_id,
        model: Arc::new(model),
    });
    
    Router::new()
        .route("/execute", post(execute::handle_execute))
        .route("/health", get(health::handle_health))
        .with_state(state)
}
