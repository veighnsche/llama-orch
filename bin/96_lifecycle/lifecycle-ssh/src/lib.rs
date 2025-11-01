//! lifecycle-ssh
//!
//! SSH-based daemon lifecycle management
//!
//! TEAM-358: Refactored from daemon-lifecycle (SSH operations only)
//! TEAM-380: Allow missing_docs for macro-generated wrapper functions
//!
//! This crate provides SSH-based remote execution for daemon lifecycle operations.
//! For local operations, use lifecycle-local crate instead.
//!
#![allow(missing_docs)] // TEAM-380: #[with_job_id] macro generates undocumented wrappers to work over SSH connections.
#![warn(clippy::all)]

pub mod install;
pub mod rebuild;
pub mod shutdown;
pub mod start;
pub mod status;
pub mod stop;
pub mod uninstall;
pub mod utils;

// TEAM-367: Re-export build types and function from shared crate
pub use lifecycle_shared::{build_daemon, BuildConfig};

// Re-export main functions
pub use install::{install_daemon, InstallConfig};
pub use rebuild::{rebuild_daemon, RebuildConfig};
pub use shutdown::{shutdown_daemon, ShutdownConfig};
pub use start::{start_daemon, HttpDaemonConfig, StartConfig};
pub use status::{check_daemon_health, DaemonStatus};
pub use stop::{stop_daemon, StopConfig};
pub use uninstall::{uninstall_daemon, UninstallConfig};

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

    // TEAM-358: Removed localhost() and is_localhost() methods
    // lifecycle-ssh should ALWAYS use SSH, even for localhost
    // If you want local operations, use lifecycle-local crate instead
}

