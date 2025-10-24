//! Daemon synchronization for declarative lifecycle management
//!
//! Created by: TEAM-280
//! Moved to shared crate: TEAM-280 (breaking change, no backward compatibility)
//!
//! This crate implements config-driven daemon synchronization for rbee.
//! Instead of imperative commands (install-hive, install-worker), users
//! define desired state in `~/.config/rbee/hives.conf` and run `rbee sync`
//! to reconcile actual state with desired state.
//!
//! # Architecture
//!
//! - **sync.rs** - Orchestrates concurrent installation across multiple hives
//! - **diff.rs** - Compares desired (config) vs actual (installed) state
//! - **install.rs** - SSH-based installation of hive and worker binaries
//! - **status.rs** - Drift detection (check if actual matches config)
//! - **validate.rs** - Config validation without applying changes
//! - **migrate.rs** - Generate config from current state
//!
//! # Key Design Decisions
//!
//! 1. **Queen installs workers via SSH** (not hive!)
//!    - Queen has global view of all hives
//!    - Can orchestrate concurrent installation
//!    - Simpler architecture (hive only manages processes)
//!
//! 2. **Concurrent installation** using tokio::spawn
//!    - 3-10x faster than sequential
//!    - Each hive installed in parallel
//!
//! 3. **Narration with job_id** for SSE routing
//!    - All operations emit events with `.job_id()`
//!    - Users see real-time progress in CLI

pub mod diff;
pub mod install;
pub mod migrate;
pub mod query; // TEAM-281: State query implementation
pub mod status;
pub mod sync;
pub mod validate;

// Re-export main functions
pub use install::{install_all, install_hive_binary, install_worker_binary};
pub use query::{query_installed_hives, query_installed_workers}; // TEAM-281
pub use status::check_status;
pub use sync::{sync_all_hives, sync_single_hive};
pub use validate::validate_config;
