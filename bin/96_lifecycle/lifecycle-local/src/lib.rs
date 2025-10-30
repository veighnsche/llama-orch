//! lifecycle-local
//!
//! LOCAL daemon lifecycle management
//!
//! TEAM-358: Refactored to remove SSH code (lifecycle-local = LOCAL only)
//!
//! This crate provides local process execution for daemon lifecycle operations.
//! For remote/SSH operations, use lifecycle-ssh crate instead.
//!
//! # Types/Utils Copied from daemon-lifecycle (REFERENCE)
//!
//! **IMPORTANT:** This crate is self-contained for LOCAL operations only.
//!
//! ## Types (inline in operation files)
//! - `start::HttpDaemonConfig` - Daemon configuration (name, health_url, args, etc.)
//!
//! ## Utils (in src/utils/)
//! - `utils::local::local_exec()` - Execute local shell commands
//! - `utils::local::local_copy()` - Copy files locally
//! - `utils::serde::*` - Serde helpers for SystemTime serialization
//!
//! ## Health Polling
//! - Uses `health-poll` crate (shared across lifecycle-local, lifecycle-ssh, lifecycle-monitored)
//!
//! ## What Each Operation Uses
//! - **build.rs** - None (uses ProcessNarrationCapture directly)
//! - **install.rs** - utils::local::{local_copy}
//! - **start.rs** - start::HttpDaemonConfig, health_poll::poll_health(), utils::local::local_exec
//! - **stop.rs** - HTTP shutdown + local process termination
//! - **shutdown.rs** - HTTP shutdown (local)
//! - **status.rs** - HTTP health check (local)
//! - **uninstall.rs** - utils::local::local_exec (file removal)
//! - **rebuild.rs** - start::HttpDaemonConfig (calls other operations)
//!
//! # Architecture
//!
//! ```text
//! rbee-keeper (CLI)
//!     ↓
//! lifecycle-local (this crate) → LOCAL daemons (queen-rbee, rbee-hive)
//!
//! For remote operations:
//! rbee-keeper → lifecycle-ssh → SSH → remote machine
//! ```
//!
//! # Design Principles
//!
//! 1. **Local execution only** - No SSH, no remote operations
//! 2. **HTTP for monitoring** - Use HTTP health checks via health-poll crate
//! 3. **Shared types** - HttpDaemonConfig used across lifecycle crates
//! 4. **Shell scripts** - Use portable shell scripts for process management
//!
//! # Module Structure
//!
//! - `start` - Start daemon locally
//! - `stop` - Stop daemon locally (HTTP + SIGTERM/SIGKILL)
//! - `status` - Check daemon status via HTTP
//! - `install` - Copy binary locally (e.g., to ~/.local/bin)
//! - `uninstall` - Remove binary locally
//! - `build` - Build binary locally
//! - `rebuild` - Rebuild and hot-reload daemon locally
//! - `shutdown` - Graceful shutdown via HTTP

#![warn(missing_docs)]
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
