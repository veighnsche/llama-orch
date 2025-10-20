//! Health endpoint for queen-rbee
//!
//! Created by: TEAM-043
//! Refactored by: TEAM-052
//! Migrated by: TEAM-164 (from http.rs to dedicated file)

use axum::{response::IntoResponse, Json};
use serde::Serialize;

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
}

/// GET /health - Health check
pub async fn handle_health() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}
