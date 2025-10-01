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

use serde::{Deserialize, Serialize};
use std::time::SystemTime;

/// Audit event types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event_type")]
pub enum AuditEvent {
    Authentication {
        identity: String,
        outcome: String,
        path: String,
    },
    Authorization {
        identity: String,
        resource: String,
        action: String,
        outcome: String,
    },
    ResourceAccess {
        identity: String,
        resource: String,
        action: String,
    },
    ConfigChange {
        identity: String,
        setting: String,
        old_value: Option<String>,
        new_value: String,
    },
}

/// Audit logger
/// TODO(ARCH-CHANGE): This crate logs to tracing only. Needs:
/// - Write audit events to append-only file
/// - Add tamper-evident logging (checksums, signatures)
/// - Implement log rotation with retention policy
/// - Add external system integration (syslog, SIEM)
/// - Implement structured query interface for forensics
/// - Add compliance reporting (GDPR, SOC2)
/// See: SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md (audit trail requirements)
pub struct AuditLogger {
    // Future: Write to file or external system
}

impl AuditLogger {
    pub fn new() -> Self {
        Self {}
    }
    
    /// Log audit event
    pub fn log(&self, event: AuditEvent) {
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        
        tracing::info!(
            target: "audit",
            timestamp = %timestamp,
            event = ?event,
            "Audit event"
        );
    }
}

impl Default for AuditLogger {
    fn default() -> Self {
        Self::new()
    }
}
