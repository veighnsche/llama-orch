//! pool-router — Request routing
//!
//! TODO(ARCH-CHANGE): This crate is a stub. Needs implementation:
//! - Implement request routing to appropriate pool
//! - Add load balancing across pools (round-robin, least-loaded)
//! - Implement sticky routing for sessions
//! - Add health-aware routing (skip unhealthy pools)
//! - Implement routing metrics
//! - Add circuit breaker integration
//! See: ARCHITECTURE_CHANGE_PLAN.md Phase 3 (pool-managerd integration)

// Medium-importance crate: TIER 3 Clippy configuration
#![warn(clippy::unwrap_used)]
#![warn(clippy::expect_used)]
#![warn(clippy::panic)]
#![warn(clippy::missing_errors_doc)]

use axum::{routing::get, Router};

pub fn create_router() -> Router {
    // TODO(ARCH-CHANGE): This is a stub router. Real implementation should:
    // - Route requests based on pool availability
    // - Implement load balancing strategies
    // - Add health checks before routing
    // - Integrate with pool-registry for pool selection
    Router::new().route("/", get(|| async { "router" }))
}
