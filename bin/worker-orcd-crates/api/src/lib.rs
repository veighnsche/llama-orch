//! worker-api — Worker HTTP API
//!
//! Exposes worker endpoints (plan, commit, ready, execute).
//!
//! # ⚠️ CRITICAL: Bearer Token Authentication
//!
//! **DO NOT HAND-ROLL TOKEN VERIFICATION**
//!
//! For Bearer token authentication middleware, use `secrets-management`:
//!
//! ```rust,ignore
//! use secrets_management::Secret;
//! use axum::{http::Request, middleware::Next, response::Response};
//!
//! async fn auth_middleware<B>(
//!     req: Request<B>,
//!     next: Next<B>,
//! ) -> Result<Response, StatusCode> {
//!     // Load expected token
//!     let expected_token = Secret::load_from_file("/etc/llorch/secrets/worker-token")?;
//!     
//!     // Extract Bearer token from header
//!     let auth_header = req.headers().get("Authorization")?;
//!     let bearer_token = auth_header.strip_prefix("Bearer ")?;
//!     
//!     // Timing-safe verification
//!     if !expected_token.verify(bearer_token) {
//!         return Err(StatusCode::UNAUTHORIZED);
//!     }
//!     
//!     Ok(next.run(req).await)
//! }
//! ```
//!
//! See: `bin/shared-crates/secrets-management/README.md`
//!
//! TODO(ARCH-CHANGE): This crate only has /ready stub. Per ARCHITECTURE_CHANGE_PLAN.md Phase 3:
//! Task Group 1 (RPC Server):
//! - Implement POST /worker/plan endpoint (feasibility check)
//! - Implement POST /worker/commit endpoint (load model into VRAM)
//! - Implement GET /worker/ready endpoint (attest worker status)
//! - Implement POST /worker/execute endpoint (run inference, stream tokens)
//! - Add Bearer token authentication middleware (use secrets-management!)
//! - Add input validation for all endpoints
//! - Add rate limiting
//! See: SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md Issue #1 (worker-orcd endpoint auth)

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
    // TODO(ARCH-CHANGE): Add all worker endpoints:
    // - .route("/worker/plan", post(plan_handler))
    // - .route("/worker/commit", post(commit_handler))
    // - .route("/worker/ready", get(ready_handler))
    // - .route("/worker/execute", post(execute_handler))
    // - .layer(auth_middleware) // Bearer token auth
    Router::new()
        .route("/ready", get(ready))
}

async fn ready() -> &'static str {
    // TODO(ARCH-CHANGE): Return proper ReadyResponse with:
    // - ready: bool
    // - handles: Vec<ModelShardHandle>
    // - nccl_group_id: Option<String> (for TP)
    "ready"
}
