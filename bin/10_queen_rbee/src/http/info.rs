//! Queen info endpoint
//!
//! TEAM-292: Added /v1/info endpoint for service discovery

use axum::Json;
use serde::Serialize;

/// Queen information response
#[derive(Debug, Serialize)]
pub struct QueenInfo {
    /// Queen's base URL
    pub base_url: String,
    /// Queen's HTTP port
    pub port: u16,
    /// Queen version
    pub version: String,
}

/// GET /v1/info - Get queen information
///
/// TEAM-292: Service discovery endpoint
/// Returns queen's address so clients (like rbee-keeper) can discover
/// where to tell hives to connect.
///
/// # Example Response
/// ```json
/// {
///   "base_url": "http://localhost:8500",
///   "port": 8500,
///   "version": "0.1.0"
/// }
/// ```
pub async fn handle_info() -> Json<QueenInfo> {
    // TEAM-292: Hardcoded for localhost-only mode
    // In the future, this could be configurable
    Json(QueenInfo {
        base_url: "http://localhost:7833".to_string(),
        port: 7833,
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_handle_info() {
        let response = handle_info().await;
        assert_eq!(response.0.base_url, "http://localhost:7833");
        assert_eq!(response.0.port, 7833);
        assert!(!response.0.version.is_empty());
    }
}
