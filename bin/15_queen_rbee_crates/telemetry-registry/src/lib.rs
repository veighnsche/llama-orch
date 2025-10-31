//! queen-rbee-telemetry-registry
//!
//! TEAM-284: Hive registry for queen-rbee
//! TEAM-374: Renamed to TelemetryRegistry - stores both hives AND workers
//!
//! Hives send telemetry (hive info + worker stats) to queen via:
//! - POST /v1/hive-heartbeat (legacy)
//! - SSE GET /v1/heartbeats/stream (new)
//!
//! This registry tracks both hive state and worker telemetry in RAM.

#![warn(missing_docs)]
#![warn(clippy::all)]

mod registry;
pub use registry::TelemetryRegistry;

// Legacy types for backward compatibility
mod types;
pub use types::HiveInfo;
