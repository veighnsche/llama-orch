// TEAM-270: Worker registry for queen-rbee

#![warn(missing_docs)]
#![warn(clippy::all)]

//! Worker registry for queen-rbee
//!
//! Workers send heartbeats directly to queen via POST /v1/worker-heartbeat.

mod registry;
pub use registry::WorkerRegistry;

// Legacy types for backward compatibility
mod types;
pub use types::{HiveRuntimeState, ResourceInfo, WorkerInfo};
