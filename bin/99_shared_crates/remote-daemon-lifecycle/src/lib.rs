//! remote-daemon-lifecycle
//!
//! Remote daemon lifecycle management via SSH
//!
//! This crate provides SSH-based remote execution for daemon lifecycle operations.
//! It wraps daemon-lifecycle functions to work over SSH connections.
//!
//! # Architecture
//!
//! ```text
//! rbee-keeper (local)
//!     ↓
//! remote-daemon-lifecycle (this crate)
//!     ↓ SSH
//! remote machine → daemon-lifecycle (local execution)
//! ```
//!
//! # Design Principles
//!
//! 1. **Minimal SSH calls** - Bundle operations into scripts when possible
//! 2. **HTTP for monitoring** - Use HTTP health checks, not SSH
//! 3. **Reuse daemon-lifecycle types** - Don't duplicate config structs
//! 4. **Shell scripts** - Use portable shell scripts for remote execution
//!
//! # Module Structure
//!
//! - `start` - Start daemon on remote machine via SSH
//! - `stop` - Stop daemon on remote machine (HTTP + SSH fallback)
//! - `status` - Check daemon status via HTTP
//! - `install` - Copy binary to remote machine via SCP
//! - `uninstall` - Remove binary from remote machine via SSH
//! - `build` - Build binary locally for remote deployment
//! - `rebuild` - Rebuild and hot-reload daemon on remote machine
//! - `shutdown` - Graceful shutdown via HTTP with SSH fallback

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod build;
pub mod install;
pub mod rebuild;
pub mod shutdown;
pub mod start;
pub mod status;
pub mod stop;
pub mod uninstall;

// Re-export main functions
pub use build::build_daemon_for_remote;
pub use install::install_daemon_remote;
pub use rebuild::rebuild_daemon_remote;
pub use shutdown::shutdown_daemon_remote;
pub use start::start_daemon_remote;
pub use status::check_daemon_status_remote;
pub use stop::stop_daemon_remote;
pub use uninstall::uninstall_daemon_remote;

/// SSH connection configuration
#[derive(Debug, Clone)]
pub struct SshConfig {
    /// Remote hostname or IP address
    pub hostname: String,
    
    /// SSH username
    pub user: String,
    
    /// SSH port (default: 22)
    pub port: u16,
}

impl SshConfig {
    /// Create new SSH config
    pub fn new(hostname: String, user: String, port: u16) -> Self {
        Self { hostname, user, port }
    }
}
