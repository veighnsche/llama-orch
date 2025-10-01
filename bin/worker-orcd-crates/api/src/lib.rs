//! worker-api — Worker HTTP API
//!
//! Exposes worker endpoints (plan, commit, ready, execute).

// High-importance crate: TIER 2 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![warn(clippy::indexing_slicing)]
#![warn(clippy::integer_arithmetic)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::missing_errors_doc)]

use axum::{routing::get, Router};

pub fn create_router() -> Router {
    Router::new()
        .route("/ready", get(ready))
}

async fn ready() -> &'static str {
    "ready"
}
