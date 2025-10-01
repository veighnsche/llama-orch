//! platform-api — Platform HTTP endpoints
//!
//! TODO(ARCH-CHANGE): This crate only has /health endpoint. Needs:
//! - Add /metrics endpoint (Prometheus format)
//! - Add /version endpoint (build info, git commit)
//! - Add /status endpoint (service health, dependencies)
//! - Add /config endpoint (safe config inspection)
//! - Add authentication middleware integration
//! - Add rate limiting per endpoint
//! See: ARCHITECTURE_CHANGE_PLAN.md Phase 4 (observability)

// Medium-importance crate: TIER 3 Clippy configuration
#![warn(clippy::unwrap_used)]
#![warn(clippy::expect_used)]
#![warn(clippy::panic)]
#![warn(clippy::missing_errors_doc)]

use axum::{routing::get, Router};

pub fn create_router() -> Router {
    // TODO(ARCH-CHANGE): Add more platform endpoints:
    // - .route("/metrics", get(metrics_handler))
    // - .route("/version", get(version_handler))
    // - .route("/status", get(status_handler))
    // - .route("/config", get(config_handler))
    Router::new().route("/health", get(|| async { "ok" }))
}
