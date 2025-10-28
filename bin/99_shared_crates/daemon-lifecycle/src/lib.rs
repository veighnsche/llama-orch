//! remote-daemon-lifecycle
//!
//! Remote daemon lifecycle management via SSH
//!
//! This crate provides SSH-based remote execution for daemon lifecycle operations.
//! It wraps daemon-lifecycle functions to work over SSH connections.
//!
//! # Types/Utils Copied from daemon-lifecycle (REFERENCE)
//!
//! **IMPORTANT:** This crate is now self-contained. The following types and utils
//! were copied from the deprecated `daemon-lifecycle` crate. DO NOT recreate them!
//!
//! ## Types (inline in operation files)
//! - `start::HttpDaemonConfig` - Daemon configuration (name, health_url, args, etc.)
//! - `utils::poll::HealthPollConfig` - Health polling configuration (base_url, max_attempts, etc.)
//!
//! ## Utils (in src/utils/)
//! - `utils::ssh::ssh_exec()` - Execute SSH commands on remote machine
//! - `utils::ssh::scp_upload()` - Upload files to remote machine via SCP
//! - `utils::poll::poll_daemon_health()` - Poll health endpoint with exponential backoff (HTTP, remote-compatible)
//! - `utils::serde::*` - Serde helpers for SystemTime serialization
//!
//! ## What Each Operation Uses
//! - **build.rs** - None (uses ProcessNarrationCapture directly)
//! - **install.rs** - utils::ssh::{ssh_exec, scp_upload}
//! - **start.rs** - start::HttpDaemonConfig, utils::poll::{poll_daemon_health, HealthPollConfig}, utils::ssh::ssh_exec
//! - **stop.rs** - utils::ssh::ssh_exec (for fallback)
//! - **shutdown.rs** - utils::ssh::ssh_exec (for fallback)
//! - **status.rs** - None (provides check_daemon_health for utils::poll)
//! - **uninstall.rs** - utils::ssh::ssh_exec
//! - **rebuild.rs** - start::HttpDaemonConfig (calls other operations)
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
pub mod utils; // TEAM-330: Utils folder (includes SSH)
               // TEAM-330: types/ folder deleted (RULE ZERO - moved inline to operation files)

// Re-export main functions
// TEAM-330: Removed "_remote" suffix - all operations are remote via SSH (RULE ZERO)
pub use build::{build_daemon, BuildConfig};
pub use install::{install_daemon, InstallConfig};
pub use rebuild::{rebuild_daemon, RebuildConfig};
pub use shutdown::{shutdown_daemon, ShutdownConfig};
pub use start::{start_daemon, HttpDaemonConfig, StartConfig}; // TEAM-330: HttpDaemonConfig moved inline
pub use status::{check_daemon_health, DaemonStatus}; // TEAM-338: RULE ZERO - Updated function signature
pub use stop::{stop_daemon, StopConfig};
pub use uninstall::{uninstall_daemon, UninstallConfig};
pub use utils::poll::HealthPollConfig; // TEAM-330: Moved from types/ to utils/poll

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

    /// Create localhost SSH config (bypasses SSH when possible)
    /// TEAM-331: Localhost mode - avoids SSH overhead for local operations
    pub fn localhost() -> Self {
        Self { hostname: "localhost".to_string(), user: whoami::username(), port: 22 }
    }

    /// Check if this config points to localhost
    /// TEAM-331: Used to bypass SSH for local operations
    pub fn is_localhost(&self) -> bool {
        self.hostname == "localhost" || self.hostname == "127.0.0.1" || self.hostname == "::1"
    }
}
