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
//! - `health` - Health check endpoint
//! - `heartbeat` - Hive heartbeat endpoint
//! - `jobs` - Job creation and SSE streaming endpoints

pub mod health;
pub mod heartbeat;
pub mod jobs;

// Re-export commonly used types
pub use health::handle_health;
pub use heartbeat::{handle_heartbeat, HeartbeatState, HttpHeartbeatAcknowledgement};
pub use jobs::{handle_create_job, handle_stream_job, HttpJobResponse, SchedulerState};
