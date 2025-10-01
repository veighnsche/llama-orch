// Pool Manager library - spawn and supervise engine processes
//
// Organized by domain:
// - core: fundamental types and state (health, registry)
// - lifecycle: spawn, drain, reload, supervision
// - placement: device masks, GPU split planning
// - validation: preflight checks
// - api: HTTP API for daemon mode
//
// ⚠️ SECURITY REMINDER: Audit Logging
//
// For security events (pool creation/deletion, node registration, policy violations),
// use `audit-logging` crate instead of hand-rolling logging:
//
// ```rust,ignore
// use audit_logging::{AuditLogger, AuditEvent};
//
// // ✅ CORRECT: Tamper-evident audit logging
// audit_logger.emit(AuditEvent::PoolCreated {
//     actor, pool_id, model_ref, node_id, replicas, gpu_devices
// }).await?;
//
// // ❌ WRONG: Regular logging (not tamper-evident)
// // tracing::info!("Pool {} created", pool_id);
// ```
//
// See: `bin/shared-crates/audit-logging/README.md`

pub mod api;
pub mod config;
pub mod core;
pub mod lifecycle;
pub mod placement;
pub mod validation;

// Re-export for backward compatibility and convenience
pub use core::{health, registry};
pub use lifecycle::{drain, preload, supervision};
pub use placement::{devicemasks, hetero_split};
pub use validation::preflight;
