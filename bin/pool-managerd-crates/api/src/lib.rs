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
//!
//! # ⚠️ AUDIT LOGGING REQUIRED
//!
//! **IMPORTANT**: Pool operations MUST be logged to `audit-logging`:
//!
//! ```rust,ignore
//! use audit_logging::{AuditLogger, AuditEvent, ActorInfo};
//!
//! // ✅ Pool creation
//! audit_logger.emit(AuditEvent::PoolCreated {
//!     timestamp: Utc::now(),
//!     actor: ActorInfo { user_id, ip, auth_method, session_id },
//!     pool_id, model_ref, node_id, replicas, gpu_devices
//! }).await?;
//!
//! // ✅ Pool deletion
//! audit_logger.emit(AuditEvent::PoolDeleted {
//!     timestamp: Utc::now(),
//!     actor: ActorInfo { user_id, ip, auth_method, session_id },
//!     pool_id, model_ref, node_id, reason
//! }).await?;
//! ```
//!
//! See: `bin/shared-crates/AUDIT_LOGGING_REMINDER.md`
//!
//! # ⚠️ INPUT VALIDATION REMINDER
//!
//! **CRITICAL**: All HTTP endpoints MUST validate inputs with `input-validation`:
//!
//! ```rust,ignore
//! use input_validation::{validate_identifier, validate_model_ref, sanitize_string};
//!
//! // Example: Validate pool_id from URL path
//! let pool_id = path.pool_id;
//! validate_identifier(&pool_id, 256)?;
//!
//! // Example: Validate model_ref from request body
//! let model_ref = body.model_ref;
//! validate_model_ref(&model_ref)?;
//!
//! // Example: Sanitize before logging
//! let safe_msg = sanitize_string(&error_msg)?;
//! log::error!("Failed to preload: {}", safe_msg);
//! ```
//!
//! **Why?** HTTP APIs are the PRIMARY attack surface
//! - ✅ Prevents injection attacks from malicious clients
//! - ✅ Validates ALL user-controlled data
//! - ✅ 253 comprehensive tests ensure safety
//!
//! See: `bin/shared-crates/input-validation/README.md`

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
