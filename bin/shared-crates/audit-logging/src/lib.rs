//! audit-logging — Security audit trail logging
//!
//! Records security-relevant events (authentication, authorization, access) for compliance and forensics.
//!
//! # Security Properties
//!
//! - Structured JSON logs
//! - Tamper-evident (append-only)
//! - Never logs secrets (uses fingerprints)
//! - Includes correlation IDs
//!
//! # ⚠️ INPUT VALIDATION REMINDER
//!
//! **Sanitize all logged data** with `input-validation`:
//!
//! ```rust,ignore
//! use input_validation::sanitize_string;
//!
//! // Sanitize before logging (prevents log injection)
//! let safe_user_id = sanitize_string(&user_id)?;
//! let safe_resource = sanitize_string(&resource_id)?;
//! audit_log.emit(AuditEvent { user: safe_user_id, resource: safe_resource });
//! ```
//!
//! **Why?** Audit logs are security-critical:
//! - ✅ Prevents ANSI escape injection
//! - ✅ Prevents control character injection
//! - ✅ Prevents Unicode directional override attacks
//!
//! See: `bin/shared-crates/input-validation/README.md`
//!
//! # Example
//!
//! ```rust
//! use audit_logging::{AuditLogger, AuditEvent};
//!
//! let logger = AuditLogger::new();
//!
//! // Log authentication event
//! logger.log(AuditEvent::Authentication {
//!     identity: "token:a3f2c1",
//!     outcome: "success",
//!     path: "/v2/tasks",
//! });
//! ```

// Security-critical crate: TIER 1 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![deny(clippy::integer_arithmetic)]
#![deny(clippy::cast_ptr_alignment)]
#![deny(clippy::mem_forget)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![warn(clippy::arithmetic_side_effects)]
#![warn(clippy::cast_lossless)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::cast_precision_loss)]
#![warn(clippy::cast_sign_loss)]
#![warn(clippy::string_slice)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::must_use_candidate)]

// Re-export public API
#[cfg(feature = "platform")]
pub use config::PlatformConfig;
pub use config::{AuditConfig, AuditMode, FlushMode, RetentionPolicy, RotationPolicy};
pub use error::AuditError;
pub use events::{ActorInfo, AuditEvent, AuditResult, AuthMethod, ResourceInfo};
pub use logger::AuditLogger;
pub use query::{AuditQuery, VerifyMode, VerifyOptions, VerifyResult};
pub use storage::AuditEventEnvelope;

// Internal modules
mod config;
mod crypto;
mod error;
mod events;
mod logger;
mod query;
mod storage;
mod writer;

// Public for BDD testing
pub mod validation;

// Optional platform mode
#[cfg(feature = "platform")]
mod platform;

#[cfg(feature = "platform")]
pub use platform::PlatformClient;
