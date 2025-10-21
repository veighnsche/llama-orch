//! HTTP endpoints for queen-rbee
//!
//! TEAM-186: Reorganized into separate modules for better organization
//!
//! ## Architecture
//!
//! All HTTP-specific code lives in this module. We keep HTTP wrappers separate
//! from business logic crates to avoid polluting pure Rust code with HTTP dependencies.
//!
//! **Pattern:**
//! - Business logic → Lives in crates (pure Rust, no HTTP)
//! - HTTP wrappers → Live HERE (thin wrappers that call crate functions)
//!
//! ## Modules
//!
//! - `lifecycle` - Hive lifecycle endpoints (start, stop, etc.)
//! - `jobs` - Job creation and management
//! - `job_stream` - SSE streaming for job execution
//! - `heartbeat` - Hive heartbeat endpoint
//! - `device_detector` - HTTP-based device detection

pub mod device_detector;
pub mod heartbeat;
pub mod job_stream;
pub mod jobs;
pub mod lifecycle;

// Re-export commonly used types
pub use device_detector::HttpDeviceDetector;
pub use heartbeat::{handle_heartbeat, HeartbeatState, HttpHeartbeatAcknowledgement};
pub use job_stream::handle_stream_job;
pub use jobs::{handle_create_job, HttpJobResponse, SchedulerState};
pub use lifecycle::{handle_hive_start, HiveStartResponse, HiveStartState};
