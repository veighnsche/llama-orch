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
pub mod health;
pub mod install;
pub mod list;
pub mod manager;
pub mod paths; // TEAM-328: Centralized path logic
pub mod rebuild; // TEAM-316: Extracted from queen-lifecycle and hive-lifecycle
pub mod shutdown;
pub mod start; // TEAM-316: Extracted from lifecycle.rs
pub mod stop; // TEAM-316: Extracted from lifecycle.rs
pub mod timeout;
pub mod uninstall;

// TEAM-259: Re-export main types and functions
// TEAM-276: Added UninstallConfig export
// TEAM-276: Added health polling with exponential backoff
// TEAM-316: HttpDaemonConfig moved to daemon-contract, lifecycle split into start/stop
// TEAM-320: Removed ensure exports (promotes explicit start/stop)
// TEAM-328: Consolidated get.rs and status.rs into health.rs (RULE ZERO)
// TEAM-328: Deleted check_daemon_status() - use check_daemon_health() directly
// TEAM-328: Removed spawn_daemon export - RULE ZERO violation (unused wrapper)
// TEAM-328: Aligned naming: verb_daemon() or verb_daemon_modifier() pattern
pub use health::{
    check_daemon_health, // TEAM-328: Renamed from is_daemon_healthy
    poll_daemon_health,  // TEAM-328: Renamed from poll_until_healthy
    HealthPollConfig,
};
pub use install::{
    build_daemon,              // TEAM-328: Build binary from source (cargo build)
    install_daemon,            // TEAM-328: Renamed from install_to_local_bin
    InstallConfig,
    InstallResult,
    UninstallConfig,
};
pub use list::{list_daemons, ListableConfig};
pub use manager::DaemonManager;
pub use rebuild::{
    rebuild_daemon, // TEAM-328: Renamed from rebuild_with_hot_reload
    RebuildConfig,
};
pub use shutdown::{
    shutdown_daemon_force,    // TEAM-328: Renamed from force_shutdown
    shutdown_daemon_graceful, // TEAM-328: Renamed from graceful_shutdown
    ShutdownConfig,
};
pub use start::start_daemon; // TEAM-328: Renamed from start_http_daemon
pub use stop::stop_daemon; // TEAM-328: Renamed from stop_http_daemon
pub use timeout::{timeout_after, with_timeout, TimeoutConfig};
pub use uninstall::uninstall_daemon;

// TEAM-316: Re-export HttpDaemonConfig from contract (it's a contract, not implementation)
pub use daemon_contract::HttpDaemonConfig;
