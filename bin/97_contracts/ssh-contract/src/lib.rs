//! ssh-contract
//!
//! TEAM-315: SSH-related contracts for rbee ecosystem
//!
//! # Purpose
//!
//! This crate provides SSH-related types used across the rbee ecosystem.
//! It eliminates duplication of `SshTarget` between ssh-config and tauri_commands.
//!
//! # Components
//!
//! - **SshTarget** - SSH host information from ~/.ssh/config
//! - **SshTargetStatus** - Connection status (online/offline/unknown)

#![warn(missing_docs)]
#![warn(clippy::all)]

/// SSH target types
pub mod target;

/// SSH target status
pub mod status;

// Re-export main types
pub use status::SshTargetStatus;
pub use target::SshTarget;
