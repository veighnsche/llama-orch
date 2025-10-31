//! HTTP endpoints for rbee-hive
//!
//! TEAM-261: Job-based architecture - ALL operations go through POST /v1/jobs
//! TEAM-339: Added shutdown endpoint
//! TEAM-372: Added SSE heartbeat stream

pub mod jobs;
pub mod shutdown; // TEAM-339: Graceful shutdown endpoint
pub mod heartbeat_stream; // TEAM-372: SSE heartbeat stream

// Re-export commonly used types
pub use shutdown::handle_shutdown; // TEAM-339: Graceful shutdown handler
pub use heartbeat_stream::{handle_heartbeat_stream, HeartbeatStreamState}; // TEAM-372
