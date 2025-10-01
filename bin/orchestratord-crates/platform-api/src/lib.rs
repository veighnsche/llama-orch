//! platform-api — Platform HTTP endpoints

// Medium-importance crate: TIER 3 Clippy configuration
#![warn(clippy::unwrap_used)]
#![warn(clippy::expect_used)]
#![warn(clippy::panic)]
#![warn(clippy::missing_errors_doc)]

use axum::{routing::get, Router};

pub fn create_router() -> Router {
    Router::new().route("/health", get(|| async { "ok" }))
}
