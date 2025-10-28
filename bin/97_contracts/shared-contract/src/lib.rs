//! shared-contract
//!
//! TEAM-284: Common contract types shared between workers and hives
//!
//! # Purpose
//!
//! This crate provides the foundation types that both `worker-contract` and
//! `hive-contract` build upon. By extracting common types here, we ensure:
//!
//! - **Consistency**: Same types used everywhere
//! - **DRY**: No duplication between worker and hive contracts
//! - **Maintainability**: Change once, applies to both
//!
//! # Architecture
//!
//! ```text
//! shared-contract (common types)
//!     ↓
//!     ├─→ worker-contract (worker-specific)
//!     └─→ hive-contract (hive-specific)
//! ```
//!
//! # What Lives Here
//!
//! - **Status types**: Health status, operational status
//! - **Heartbeat traits**: Common heartbeat behavior
//! - **Timestamp handling**: Consistent time representation
//! - **Error types**: Shared error definitions
//! - **Constants**: Timeouts, intervals, etc.

#![warn(missing_docs)]
#![warn(clippy::all)]

/// Status types (health, operational state)
pub mod status;

/// Heartbeat protocol definitions
pub mod heartbeat;

/// Common error types
pub mod error;

/// Shared constants (timeouts, intervals)
pub mod constants;

// Re-export commonly used types
pub use error::ContractError;
pub use heartbeat::{HeartbeatPayload, HeartbeatTimestamp};
pub use status::{HealthStatus, OperationalStatus};

// Re-export constants
pub use constants::{
    CLEANUP_INTERVAL_SECS, HEARTBEAT_INTERVAL_SECS, HEARTBEAT_TIMEOUT_SECS, MAX_HEARTBEAT_AGE_SECS,
};
