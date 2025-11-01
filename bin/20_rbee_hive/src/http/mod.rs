//! HTTP endpoints for rbee-hive
//!
//! TEAM-261: Job-based architecture - ALL operations go through POST /v1/jobs
//! TEAM-339: Added shutdown endpoint
//! TEAM-372: Added SSE heartbeat stream
//! TEAM-374: Added dev proxy for Vite dev server
//! TEAM-378: Added static file serving for production UI

pub mod jobs;
pub mod shutdown; // TEAM-339: Graceful shutdown endpoint
pub mod heartbeat_stream; // TEAM-372: SSE heartbeat stream
pub mod dev_proxy; // TEAM-374: Development proxy
pub mod static_files; // TEAM-378: Static file serving

// Re-export commonly used types
pub use shutdown::handle_shutdown; // TEAM-339: Graceful shutdown handler
pub use heartbeat_stream::handle_heartbeat_stream; // TEAM-372
pub use dev_proxy::{dev_proxy_handler, DevProxyState}; // TEAM-378: Development proxy handler with state
pub use static_files::create_static_router; // TEAM-378: Static file router
