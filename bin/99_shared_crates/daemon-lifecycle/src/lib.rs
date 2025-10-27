// TEAM-135: Created by TEAM-135 (scaffolding)
// TEAM-152: Implemented core daemon spawning functionality
// TEAM-152: Replaced tracing with narration for observability
// TEAM-259: Split into modules for better organization
// TEAM-259: Added CRUD operations (install, list, get, status)
// Purpose: Shared daemon lifecycle management for rbee-keeper, queen-rbee, and rbee-hive

#![warn(missing_docs)]
#![warn(clippy::all)]

//! daemon-lifecycle
//!
//! **Category:** Utility
//! **Pattern:** Function-based
//! **Standard:** See `/bin/CRATE_INTERFACE_STANDARD.md`
//!
//! Shared daemon lifecycle management functionality for managing daemon processes
//! across rbee-keeper, queen-rbee, and rbee-hive binaries.
//! All observability is handled through narration-core (no tracing).
//!
//! # Module Structure
//!
//! - `manager` - DaemonManager for spawning daemon processes
//! - `health` - HTTP health checking for daemons
//! - `install` - Install/uninstall daemon binaries (TEAM-259)
//! - `list` - List all daemon instances (TEAM-259)
//! - `get` - Get daemon instance by ID (TEAM-259)
//! - `status` - Check daemon status (TEAM-259)
//!
//! # Interface
//!
//! ## Daemon Spawning
//! ```rust,no_run
//! use daemon_lifecycle::{DaemonManager, spawn_daemon};
//! use std::path::PathBuf;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Create daemon manager
//! let manager = DaemonManager::new(
//!     PathBuf::from("target/debug/queen-rbee"),
//!     vec!["--config".to_string(), "config.toml".to_string()]
//! );
//!
//! // Spawn daemon process
//! let child = manager.spawn().await?;
//!
//! // Find binary in target directory
//! let binary_path = DaemonManager::find_in_target("queen-rbee")?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Health Checking
//! ```rust,no_run
//! use daemon_lifecycle::is_daemon_healthy;
//!
//! # async fn example() {
//! let is_healthy = is_daemon_healthy(
//!     "http://localhost:8500",
//!     None,  // Use default /health endpoint
//!     None,  // Use default 2s timeout
//! ).await;
//! # }
//! ```
//!
//! ## Health Polling with Exponential Backoff (TEAM-276)
//! ```rust,no_run
//! use daemon_lifecycle::{poll_until_healthy, HealthPollConfig};
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Poll until daemon is healthy (for startup synchronization)
//! let config = HealthPollConfig::new("http://localhost:8500")
//!     .with_max_attempts(10)
//!     .with_job_id("job-123")
//!     .with_daemon_name("queen-rbee");
//!
//! poll_until_healthy(config).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Timeout Enforcement (TEAM-276)
//! ```rust,no_run
//! use daemon_lifecycle::{with_timeout, TimeoutConfig};
//! use std::time::Duration;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = TimeoutConfig::new("fetch_data", Duration::from_secs(30))
//!     .with_job_id("job-123");
//!
//! let result = with_timeout(config, async {
//!     // Your operation here
//!     Ok(42)
//! }).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Graceful Shutdown (TEAM-276)
//! ```rust,no_run
//! use daemon_lifecycle::{graceful_shutdown, ShutdownConfig};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = ShutdownConfig::new(
//!     "queen-rbee",
//!     "http://localhost:8500",
//!     "http://localhost:8500/v1/shutdown",
//! ).with_job_id("job-123");
//!
//! graceful_shutdown(config).await?;
//! # Ok(())
//! # }
//! ```
//!
//! ## High-Level Lifecycle (TEAM-276)
//! ```rust,no_run
//! use daemon_lifecycle::{start_http_daemon, stop_http_daemon};
//! use daemon_contract::HttpDaemonConfig;
//! use std::path::PathBuf;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Start daemon (spawn + health poll)
//! let config = HttpDaemonConfig::new(
//!     "queen-rbee",
//!     PathBuf::from("target/release/queen-rbee"),
//!     "http://localhost:8500",
//! ).with_job_id("job-123");
//!
//! let child = start_http_daemon(config.clone()).await?;
//! std::mem::forget(child); // Keep daemon alive
//!
//! // Stop daemon gracefully
//! stop_http_daemon(config).await?;
//! # Ok(())
//! # }
//! ```

// TEAM-259: Module declarations
// TEAM-276: Added high-level lifecycle operations
// TEAM-316: Split lifecycle into start/stop modules (RULE ZERO - single responsibility)
// TEAM-320: Removed ensure module (promotes explicit start/stop)
// TEAM-328: Deleted get.rs and status.rs - consolidated into health.rs (RULE ZERO)
// TEAM-328: Added paths module - centralized path constants to ensure install/uninstall consistency
// TEAM-329: Renamed manager → find, extracted build from install (single responsibility)
// TEAM-329: Inlined daemon-contract into types/ (RULE ZERO - 1 consumer = inline it)
// TEAM-329: Moved find + timeout + paths + poll to utils/, extracted ALL config types to types/
pub mod build; // TEAM-329: Extracted from install.rs
pub mod install;
pub mod rebuild; // TEAM-316: Extracted from queen-lifecycle and hive-lifecycle
pub mod shutdown;
pub mod start; // TEAM-316: Extracted from lifecycle.rs
pub mod status; // TEAM-329: Renamed from health.rs (checking status, not health)
pub mod stop; // TEAM-316: Extracted from lifecycle.rs
pub mod types; // TEAM-329: ALL config types, types/{op}.rs matches src/{op}.rs
pub mod uninstall;
pub mod utils; // TEAM-329: Utilities (find, paths, poll, timeout)

// TEAM-329: DELETED list.rs - UNUSED (exported but never implemented, zero consumers)

// TEAM-259: Re-export main types and functions
// TEAM-276: Added UninstallConfig export
// TEAM-276: Added health polling with exponential backoff
// TEAM-316: HttpDaemonConfig moved to daemon-contract, lifecycle split into start/stop
// TEAM-320: Removed ensure exports (promotes explicit start/stop)
// TEAM-328: Consolidated get.rs and status.rs into health.rs (RULE ZERO)
// TEAM-328: Deleted check_daemon_status() - use check_daemon_health() directly
// TEAM-328: Removed spawn_daemon export - RULE ZERO violation (unused wrapper)
// TEAM-328: Aligned naming: verb_daemon() or verb_daemon_modifier() pattern
// TEAM-329: All config types re-exported from types module
pub use build::build_daemon; // TEAM-329: Extracted from install.rs to build.rs
pub use install::install_daemon; // TEAM-328: Renamed from install_to_local_bin
pub use rebuild::rebuild_daemon; // TEAM-329: Renamed from update_daemon (user request)
pub use shutdown::shutdown_daemon; // TEAM-329: Simplified from shutdown_daemon_force
pub use start::start_daemon; // TEAM-328: Renamed from start_http_daemon
pub use status::check_daemon_health; // TEAM-329: Renamed from health.rs to status.rs
pub use stop::stop_daemon; // TEAM-328: Renamed from stop_http_daemon
pub use types::{
    // TEAM-329: All config types from types/ module (PERFECT PARITY)
    HealthPollConfig,
    HttpDaemonConfig,
    InstallConfig,
    InstallResult,
    RebuildConfig,
    ShutdownConfig,
    StatusRequest,
    StatusResponse,
    TimeoutConfig,
    UninstallConfig,
};
pub use uninstall::uninstall_daemon;
pub use utils::{
    // TEAM-329: Utilities from utils/ module
    find_binary,
    get_install_dir,
    get_install_path,
    get_pid_file_path,  // TEAM-329: Centralized in utils/pid.rs
    poll_daemon_health, // TEAM-329: Moved from health.rs to utils/poll.rs
    read_pid_file,      // TEAM-329: Centralized in utils/pid.rs
    remove_pid_file,    // TEAM-329: Centralized in utils/pid.rs
    timeout_after,
    with_timeout,
    write_pid_file, // TEAM-329: Centralized in utils/pid.rs
};
