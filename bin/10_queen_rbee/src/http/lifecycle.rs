//! Hive lifecycle HTTP endpoints
//!
//! TEAM-186: HTTP wrappers for hive lifecycle operations

use axum::{extract::State, http::StatusCode, Json};
use queen_rbee_hive_catalog::HiveCatalog;
use queen_rbee_hive_lifecycle;
use serde::Serialize;
use std::sync::Arc;

/// Response for hive start operation
#[derive(Debug, Serialize)]
pub struct HiveStartResponse {
    /// URL of the started hive
    pub hive_url: String,
    /// ID of the hive
    pub hive_id: String,
    /// Port the hive is running on
    pub port: u16,
}

/// State for hive start endpoint
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
