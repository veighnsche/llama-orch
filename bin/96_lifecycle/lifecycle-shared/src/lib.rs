//! lifecycle-shared
//!
//! Shared utilities for lifecycle management (local and remote)
//!
//! TEAM-367: Created to eliminate duplication between lifecycle-local and lifecycle-remote
//! TEAM-377: Added BINARY_INSTALL_DIR constant to eliminate path duplication
//!
//! # Shared Components
//!
//! ## Types
//! - `HttpDaemonConfig` - Daemon configuration (name, health_url, args, etc.)
//! - `DaemonStatus` - Daemon status information (is_running, is_installed)
//! - `BuildConfig` - Build configuration (daemon_name, target, job_id, features)
//!
//! ## Utils
//! - `utils::serde` - Serde helpers for SystemTime serialization
//!
//! # Architecture
//!
//! ```text
//! lifecycle-local ──┐
//!                   ├──> lifecycle-shared (this crate)
//! lifecycle-ssh  ───┘
//! ```
//!
//! # Design Principles
//!
//! 1. **Shared types only** - No execution logic, just data structures
//! 2. **Minimal dependencies** - Only what's needed for types
//! 3. **No SSH or local-specific code** - Pure shared abstractions

#![warn(missing_docs)]
#![warn(clippy::all)]

// TEAM-377: RULE ZERO - Single source of truth for binary installation directory
/// Directory where daemon binaries are installed (relative to $HOME)
///
/// Used by install, uninstall, start, and status operations.
/// MUST be consistent across all lifecycle operations.
pub const BINARY_INSTALL_DIR: &str = ".local/bin";

pub mod build;
pub mod install;
pub mod start;
pub mod status;
pub mod utils;

// Re-export main types
pub use build::{build_daemon, BuildConfig};
pub use install::resolve_binary_path;
pub use start::{build_start_command, find_binary_command, HttpDaemonConfig};
pub use status::{check_health_http, normalize_health_url, DaemonStatus};
