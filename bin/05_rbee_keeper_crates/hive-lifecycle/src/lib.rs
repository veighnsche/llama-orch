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
//! start_hive("gpu-server", "/usr/local/bin", 9000).await?;
//! ```

#![warn(missing_docs)]

// TEAM-290: Individual files for each operation (matches lifecycle crate pattern)
pub mod ssh;
pub mod install;
pub mod uninstall;
pub mod start;
pub mod stop;
pub mod status;

// Re-export for convenience
pub use ssh::SshClient;
pub use install::install_hive;
pub use uninstall::uninstall_hive;
pub use start::start_hive;
pub use stop::stop_hive;
pub use status::{is_hive_running, hive_status};
