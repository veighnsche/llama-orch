//! Queen info endpoint
//!
//! TEAM-292: Added /v1/info endpoint for service discovery
//! TEAM-CLEANUP: Consolidated /v1/build-info into /v1/info (Rule Zero)

use axum::Json;
use serde::Serialize;

/// Queen information response
///
/// TEAM-CLEANUP: Consolidated build info and service discovery into one endpoint
#[derive(Debug, Serialize)]
pub struct QueenInfo {
    /// Queen's base URL
    pub base_url: String,
    /// Queen's HTTP port
    pub port: u16,
    /// Queen version
    pub version: String,
    /// List of enabled Cargo features (e.g., "local-hive")
    pub features: Vec<String>,
    /// Build timestamp (from BUILD_TIMESTAMP env var, or "unknown")
    pub build_timestamp: String,
}

/// GET /v1/info - Get queen information
///
/// TEAM-292: Service discovery endpoint
/// TEAM-CLEANUP: Consolidated /v1/build-info into this endpoint (Rule Zero)
///
/// Returns queen's address, version, features, and build info.
/// Used for:
/// - Service discovery (where is queen located?)
/// - Capability detection (what features does queen have?)
///
/// # Example Response
/// ```json
/// {
///   "base_url": "http://localhost:7833",
///   "port": 7833,
///   "version": "0.1.0",
///   "features": ["local-hive"],
///   "build_timestamp": "2025-10-29T14:00:00Z"
/// }
/// ```
pub async fn handle_info() -> Json<QueenInfo> {
    let features = {
        let mut f = vec![];

        #[cfg(feature = "local-hive")]
        f.push("local-hive".to_string());
        f
    };

    // TEAM-292: Hardcoded for localhost-only mode
    // In the future, this could be configurable
    Json(QueenInfo {
        base_url: "http://localhost:7833".to_string(),
        port: 7833,
        version: env!("CARGO_PKG_VERSION").to_string(),
        features,
        build_timestamp: option_env!("BUILD_TIMESTAMP").unwrap_or("unknown").to_string(),
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
