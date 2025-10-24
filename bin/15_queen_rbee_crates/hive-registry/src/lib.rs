//! queen-rbee-hive-registry
//!
//! TEAM-284: Hive registry for queen-rbee
//!
//! Hives send heartbeats directly to queen via POST /v1/hive-heartbeat.
//! This registry tracks hive state in RAM based on those heartbeats.
//!
//! Mirrors `worker-registry` but for hives.

#![warn(missing_docs)]
#![warn(clippy::all)]

mod registry;
pub use registry::HiveRegistry;

// Legacy types for backward compatibility
mod types;
pub use types::HiveInfo;
