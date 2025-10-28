//! HTTP endpoints for rbee-hive
//!
//! TEAM-261: Job-based architecture - ALL operations go through POST /v1/jobs
//! TEAM-339: Added shutdown endpoint

pub mod jobs;
pub mod shutdown; // TEAM-339: Graceful shutdown endpoint

// Re-export commonly used types
pub use shutdown::handle_shutdown; // TEAM-339: Graceful shutdown handler
