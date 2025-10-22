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

// TEAM-259: Module declarations
pub mod ensure;
pub mod get;
pub mod health;
pub mod install;
pub mod list;
pub mod manager;
pub mod status;

// TEAM-259: Re-export main types and functions
pub use ensure::ensure_daemon_running;
pub use get::{get_daemon, GettableConfig};
pub use health::is_daemon_healthy;
pub use install::{install_daemon, uninstall_daemon, InstallConfig, InstallResult};
pub use list::{list_daemons, ListableConfig};
pub use manager::{spawn_daemon, DaemonManager};
pub use status::{check_daemon_status, StatusRequest, StatusResponse};
