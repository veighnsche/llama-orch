//! pool-managerd-api — HTTP API for pool-managerd
//!
//! Exposes pool management endpoints.
//!
//! TODO(ARCH-CHANGE): This crate only has /health endpoint. Per ARCHITECTURE_CHANGE_PLAN.md:
//! - Add POST /pools/:id/preload endpoint (model preloading)
//! - Add GET /pools/:id/status endpoint (pool health and capacity)
//! - Add POST /pools/:id/drain endpoint (graceful shutdown)
//! - Add POST /pools/:id/reload endpoint (config reload)
//! - Add POST /workers/register endpoint (worker registration)
//! - Add Bearer token authentication middleware
//! - Add rate limiting per endpoint
//! See: SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md Issue #8 (pool-managerd auth)

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
    // TODO(ARCH-CHANGE): Add all pool-managerd endpoints:
    // - .route("/pools/:id/preload", post(preload_handler))
    // - .route("/pools/:id/status", get(status_handler))
    // - .route("/pools/:id/drain", post(drain_handler))
    // - .route("/pools/:id/reload", post(reload_handler))
    // - .route("/workers/register", post(register_worker_handler))
    // - .layer(auth_middleware) // Bearer token auth
    Router::new()
        .route("/health", get(health))
}

async fn health() -> &'static str {
    // TODO(ARCH-CHANGE): Return proper health response with:
    // - pools: Vec<PoolHealth>
    // - workers: Vec<WorkerHealth>
    // - system: SystemHealth (CPU, memory, disk)
    "ok"
}
