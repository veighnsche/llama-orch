//! hive-contract
//!
//! TEAM-284: Contract definition for hive implementations in the rbee system
//!
//! # Overview
//!
//! This crate defines the types and protocols that ALL hives must implement.
//! Mirrors `worker-contract` but for hives.
//!
//! # Hive Lifecycle
//!
//! ```text
//! Queen spawns hive → Hive starts → Hive reports ready → Hive manages workers
//!                                ↓
//!                        Heartbeat every 30s to queen
//! ```
//!
//! # Key Concepts
//!
//! - **HiveInfo**: Complete hive state (hostname, port, capabilities, etc.)
//! - **HiveHeartbeat**: Periodic status update sent to queen
//! - **HiveStatus**: Current hive state (uses shared OperationalStatus)
//! - **Hive HTTP API**: Endpoints all hives must implement
//!
//! # Example
//!
//! ```no_run
//! use hive_contract::{HiveInfo, HiveHeartbeat};
//! use shared_contract::{OperationalStatus, HealthStatus};
//! use chrono::Utc;
//!
//! // Hive creates its info
//! let hive = HiveInfo {
//!     id: "localhost".to_string(),
//!     hostname: "127.0.0.1".to_string(),
//!     port: 9200,
//!     operational_status: OperationalStatus::Ready,
//!     health_status: HealthStatus::Healthy,
//!     version: "0.1.0".to_string(),
//!     // TODO: Add system stats (CPU, RAM, VRAM, temperature)
//! };
//!
//! // Hive sends heartbeat to queen
//! let heartbeat = HiveHeartbeat::new(hive);
//!
//! // Send to queen: POST http://queen:8500/v1/hive-heartbeat
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

/// Hive contract types
pub mod types;

/// Heartbeat protocol for hives
pub mod heartbeat;

/// Hive HTTP API specification
pub mod api;

/// Telemetry types (TEAM-381: moved from rbee-hive-monitor)
pub mod telemetry;

// Re-export main types for convenience
pub use heartbeat::{HiveDevice, HiveHeartbeat};
pub use telemetry::{HeartbeatSnapshot, HiveHeartbeatEvent, HiveTelemetry, ProcessStats, QueenHeartbeat};
pub use types::HiveInfo;

// Re-export shared types
pub use shared_contract::{
    HealthStatus, HeartbeatTimestamp, OperationalStatus, HEARTBEAT_INTERVAL_SECS,
    HEARTBEAT_TIMEOUT_SECS,
};
