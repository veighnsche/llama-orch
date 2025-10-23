//! Build information endpoint
//!
//! TEAM-262: Added /v1/build-info endpoint for rbee-keeper to query

use axum::Json;
use serde::Serialize;

#[derive(Serialize)]
pub struct BuildInfo {
    pub version: String,
    pub features: Vec<String>,
    pub build_timestamp: String,
}

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
