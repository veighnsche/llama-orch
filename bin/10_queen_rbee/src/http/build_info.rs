//! Build information endpoint
//!
//! Created by: TEAM-262
//! Purpose: Allow rbee-keeper to query queen's build configuration
//! Endpoint: GET /v1/build-info
//!
//! # Usage
//!
//! ```bash
//! curl http://localhost:7833/v1/build-info
//! ```
//!
//! # Response
//!
//! ```json
//! {
//!   "version": "0.1.0",
//!   "features": ["local-hive"],
//!   "build_timestamp": "2025-10-23T11:00:00Z"
//! }
//! ```

use axum::Json;
use serde::Serialize;

/// Build information response
///
/// Contains version, enabled features, and build timestamp.
/// Used by rbee-keeper to determine queen's capabilities.
///
/// # Fields
///
/// * `version` - Cargo package version
/// * `features` - List of enabled Cargo features
/// * `build_timestamp` - When the binary was built (if available)
///
/// # Example
///
/// ```rust,ignore
/// let info = BuildInfo {
///     version: "0.1.0".to_string(),
///     features: vec!["local-hive".to_string()],
///     build_timestamp: "2025-10-23T11:00:00Z".to_string(),
/// };
/// ```
#[derive(Serialize)]
pub struct BuildInfo {
    /// Cargo package version (from CARGO_PKG_VERSION)
    pub version: String,
    /// List of enabled Cargo features (e.g., "local-hive")
    pub features: Vec<String>,
    /// Build timestamp (from BUILD_TIMESTAMP env var, or "unknown")
    pub build_timestamp: String,
}

/// Handle GET /v1/build-info endpoint
///
/// Returns queen's build configuration including version, features, and timestamp.
/// This allows rbee-keeper to query queen's capabilities and suggest optimizations.
///
/// # Returns
///
/// JSON response with build information
///
/// # Example
///
/// ```rust,ignore
/// use axum::Router;
/// use axum::routing::get;
///
/// let app = Router::new()
///     .route("/v1/build-info", get(handle_build_info));
/// ```
///
/// # TEAM-262
///
/// Created to support smart prompts for localhost optimization.
/// rbee-keeper queries this endpoint before installing localhost hive
/// to check if queen has the `local-hive` feature enabled.
pub async fn handle_build_info() -> Json<BuildInfo> {
    let mut features = vec![];

    #[cfg(feature = "local-hive")]
    features.push("local-hive".to_string());

    Json(BuildInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        features,
        build_timestamp: option_env!("BUILD_TIMESTAMP").unwrap_or("unknown").to_string(),
    })
}
