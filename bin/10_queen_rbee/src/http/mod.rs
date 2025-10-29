//! HTTP endpoints for queen-rbee
//!
//! TEAM-186: Reorganized into separate modules for better organization
//! TEAM-217: Investigated Oct 22, 2025 - Behavior inventory complete
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
//! - `heartbeat_stream` - SSE endpoint for live heartbeat updates (TEAM-285)
//! - `jobs` - Job creation and SSE streaming endpoints
//! - `build_info` - Build information endpoint (TEAM-262)
//! - `shutdown` - Graceful shutdown endpoint (TEAM-339)

// TEAM-CLEANUP: Deleted build_info module - consolidated into info.rs (Rule Zero)
pub mod health;
pub mod heartbeat;
pub mod heartbeat_stream; // TEAM-285: Live heartbeat streaming
pub mod info; // TEAM-292: Queen info endpoint for service discovery
pub mod jobs; // TEAM-262
pub mod shutdown; // TEAM-339: Graceful shutdown endpoint
pub mod static_files; // TEAM-293: Static file serving for web UI

// Re-export commonly used types
// TEAM-CLEANUP: Deleted build_info - consolidated into info.rs (Rule Zero)
pub use health::handle_health;
pub use heartbeat::{
    handle_hive_heartbeat, // TEAM-284/285: Hive heartbeat handler
    handle_worker_heartbeat,
    HeartbeatState,
    HttpHeartbeatAcknowledgement, // TEAM-275: Removed handle_heartbeat (deprecated)
};
pub use heartbeat_stream::handle_heartbeat_stream; // TEAM-285: Live heartbeat streaming
pub use info::handle_info; // TEAM-292: Queen info handler
pub use jobs::{handle_cancel_job, handle_create_job, handle_stream_job, SchedulerState}; // TEAM-262, TEAM-305-FIX: Added handle_cancel_job
pub use shutdown::handle_shutdown; // TEAM-339: Graceful shutdown handler
pub use static_files::create_static_router; // TEAM-293: Static file serving for web UI
