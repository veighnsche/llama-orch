//! GET /health endpoint - Health check

use crate::http::AppState;
use axum::{extract::State, Json};
use serde::Serialize;
use std::sync::Arc;

#[derive(Serialize)]
pub struct HealthResponse {
    status: String,
    vram_bytes: u64,
    uptime_seconds: u64,
}

/// Handle GET /health
pub async fn handle_health(
    State(state): State<Arc<AppState>>,
) -> Json<HealthResponse> {
    // Check VRAM residency
    let vram_resident = state.model.check_vram_residency().unwrap_or(false);
    
    let status = if vram_resident {
        "healthy"
    } else {
        "unhealthy"
    };
    
    Json(HealthResponse {
        status: status.to_string(),
        vram_bytes: state.model.vram_bytes(),
        uptime_seconds: 0, // TODO: track startup time
    })
}
