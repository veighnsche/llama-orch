//! platform-api — HTTP API for llama-orchestrator
//!
//! Exposes REST endpoints for model inference, session management, and platform operations.
//!
//! # ⚠️ SECURITY: API Token Authentication
//!
//! For Bearer token authentication, use `secrets-management`:
//!
//! ```rust,ignore
//! use secrets_management::Secret;
//!
//! let api_token = Secret::load_from_file("/etc/llorch/secrets/api-token")?;
//! if !api_token.verify(bearer_token) {
//!     return Err(StatusCode::UNAUTHORIZED);
//! }
//! ```
//!
//! See: `bin/shared-crates/secrets-management/README.md`
//! See: ARCHITECTURE_CHANGE_PLAN.md Phase 4 (observability)
//!
//! # ⚠️ INPUT VALIDATION REMINDER
//!
//! **Always validate HTTP inputs** with `input-validation` crate:
//!
//! ```rust,ignore
//! use input_validation::{validate_identifier, validate_range, sanitize_string};
//!
//! // Validate query parameters
//! validate_identifier(&query.service_name, 256)?;
//! validate_range(query.limit, 1, 1000)?;
//!
//! // Sanitize before logging
//! let safe_input = sanitize_string(&user_input)?;
//! ```
//!
//! See: `bin/shared-crates/input-validation/README.md`

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
