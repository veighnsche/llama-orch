//! Health check HTTP endpoint
//!
//! TEAM-217: Investigated Oct 22, 2025 - Behavior inventory complete
//!
//! Simple health check for queen-rbee

use axum::http::StatusCode;

/// GET /health - Health check endpoint
///
/// Returns 200 OK if queen-rbee is running
pub async fn handle_health() -> StatusCode {
    StatusCode::OK
}
