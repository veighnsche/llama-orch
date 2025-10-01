//! # Security Reminder: Audit Logging
//!
//! **IMPORTANT**: For security-critical events (authentication, authorization, access control),
//! use the `audit-logging` crate instead of hand-rolling your own logging.
//!
//! ```rust,ignore
//! use audit_logging::{AuditLogger, AuditEvent};
//!
//! // ✅ CORRECT: Use audit-logging for security events
//! audit_logger.emit(AuditEvent::AuthSuccess { /* ... */ }).await?;
//!
//! // ❌ WRONG: Don't hand-roll security logging
//! // tracing::info!("User {} authenticated", user_id); // Not tamper-evident!
//! ```
//!
//! **Why?**
//! - ✅ Tamper-evident (SHA-256 hash chains)
//! - ✅ Input validation (prevents log injection)
//! - ✅ SOC2/GDPR compliant
//! - ✅ Secure file permissions
//! - ✅ Comprehensive test coverage (85%)
//!
//! See: `bin/shared-crates/audit-logging/README.md`

pub mod admission;
pub mod api;
pub mod app;
pub mod clients;
pub mod config;
pub mod domain;
pub mod infra;
pub mod metrics;
pub mod ports;
pub mod services;
pub mod state;
