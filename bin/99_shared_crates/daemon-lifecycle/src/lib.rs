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
//! - `ensure` - "Ensure daemon running" pattern (TEAM-259)
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
//! ## Ensure Daemon Running
//! ```rust,no_run
//! use daemon_lifecycle::ensure_daemon_running;
//! use anyhow::Result;
//!
//! # async fn example() -> Result<()> {
//! let was_running = ensure_daemon_running(
//!     "queen-rbee",
//!     "http://localhost:8500",
//!     None,  // No job_id
//!     || async {
//!         // Spawn daemon here
//!         Ok(())
//!     },
//!     None,  // Default 30s timeout
//!     None,  // Default 500ms poll interval
//! ).await?;
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
//! use daemon_lifecycle::{start_http_daemon, stop_http_daemon, HttpDaemonConfig};
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
// TEAM-276: Added timeout, shutdown, and lifecycle modules
pub mod ensure;
pub mod get;
pub mod health;
pub mod install;
pub mod lifecycle;
pub mod list;
pub mod manager;
pub mod shutdown;
pub mod status;
pub mod timeout;

// TEAM-259: Re-export main types and functions
// TEAM-276: Added UninstallConfig export
// TEAM-276: Added health polling with exponential backoff
// TEAM-276: Added timeout enforcement and graceful shutdown
// TEAM-276: Added high-level lifecycle operations
// TEAM-276: Added ensure pattern with handle support
pub use ensure::{ensure_daemon_running, ensure_daemon_with_handle};
pub use get::{get_daemon, GettableConfig};
pub use health::{is_daemon_healthy, poll_until_healthy, HealthPollConfig};
pub use install::{install_daemon, uninstall_daemon, InstallConfig, InstallResult, UninstallConfig};
pub use lifecycle::{start_http_daemon, stop_http_daemon, HttpDaemonConfig};
pub use list::{list_daemons, ListableConfig};
pub use manager::{spawn_daemon, DaemonManager};
pub use shutdown::{force_shutdown, graceful_shutdown, ShutdownConfig};
pub use status::{check_daemon_status, StatusRequest, StatusResponse};
pub use timeout::{timeout_after, with_timeout, TimeoutConfig};
