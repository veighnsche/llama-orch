//! Hive Lifecycle Management for rbee-keeper
//!
//! TEAM-290: Remote hive operations via SSH
//!
//! This crate provides lifecycle management for rbee-hive instances:
//! - Install/uninstall hives remotely via SSH
//! - Start/stop hives remotely
//! - Health checks
//! - Auto-update
//!
//! # Design Philosophy
//!
//! **Combines SSH + daemon-lifecycle:**
//! - Uses ssh-client for remote operations
//! - Uses daemon-lifecycle for lifecycle patterns
//! - Piggybacks on host ~/.ssh/config
//! - No custom SSH config format
//!
//! # Example
//!
//! ```rust,ignore
//! use hive_lifecycle::{install_hive, start_hive};
//!
//! // User has this in ~/.ssh/config:
//! // Host gpu-server
//! //   HostName 192.168.1.100
//! //   User ubuntu
//! //   IdentityFile ~/.ssh/id_rsa
//!
//! // Install hive remotely
//! install_hive("gpu-server", "./rbee-hive", "/usr/local/bin").await?;
//!
//! // Start hive remotely
//! start_hive("gpu-server", "/usr/local/bin", 7835, None).await?;
//! ```

#![warn(missing_docs)]

// TEAM-314: Default install directory (matches queen-lifecycle)
// Both queen and hive use ~/.local/bin (no sudo needed)
/// Default installation directory for hive binary
pub const DEFAULT_INSTALL_DIR: &str = "$HOME/.local/bin";

// TEAM-314: Default build directory for remote builds
/// Default build directory for remote hive builds
pub const DEFAULT_BUILD_DIR: &str = "/tmp/llama-orch-build";

// TEAM-290: Individual files for each operation (matches lifecycle crate pattern)
pub mod install;
pub mod uninstall;
pub mod start;
pub mod stop;
pub mod status;
pub mod rebuild; // TEAM-314: Added for parity with queen

// TEAM-314: Re-export SshClient from shared ssh-config crate
pub use ssh_config::SshClient;

// Re-export main functions
pub use install::install_hive;
pub use uninstall::uninstall_hive;
pub use start::start_hive;
pub use stop::stop_hive;
pub use rebuild::rebuild_hive; // TEAM-314: Added for parity with queen

// TEAM-314: Export status functions (if they exist)
#[cfg(feature = "status")]
pub use status::{is_hive_running, hive_status};
