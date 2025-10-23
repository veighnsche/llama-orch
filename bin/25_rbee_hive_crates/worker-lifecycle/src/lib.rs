// TEAM-271: Worker lifecycle implementation using daemon-lifecycle
#![warn(missing_docs)]
#![warn(clippy::all)]

//! rbee-hive-worker-lifecycle
//!
//! **Category:** Orchestration
//! **Pattern:** Command Pattern
//! **Standard:** See `/bin/CRATE_INTERFACE_STANDARD.md`
//!
//! Lifecycle management for LLM worker instances.
//! Uses daemon-lifecycle for process spawning and worker-catalog for binary resolution.
//!
//! # TEAM-277 Architecture Update
//!
//! Worker installation is now handled by queen-rbee via SSH (see package_manager module).
//! Hive only manages worker PROCESSES (start/stop/list/get).
//! The install/uninstall modules are API stubs for consistency.
//!
//! # Module Structure
//!
//! TEAM-276: Standardized file naming for consistency across lifecycle crates
//!
//! - `types` - Request/Response types for all operations
//! - `start` - Start worker operations (TEAM-271, renamed from spawn)
//! - `stop` - Stop worker operations (TEAM-272, renamed from delete)
//! - `list` - List worker processes (TEAM-274, renamed from process_list)
//! - `get` - Get worker process details (TEAM-274, renamed from process_get)
//! - `install` - Install worker binary (TEAM-276, stub for consistency)
//! - `uninstall` - Uninstall worker binary (TEAM-276, stub for consistency)
//!
//! # Interface
//!
//! ## Worker Spawning
//! ```rust,no_run
//! use rbee_hive_worker_lifecycle::{spawn_worker, WorkerSpawnConfig};
//!
//! # async fn example() -> anyhow::Result<()> {
//! let config = WorkerSpawnConfig {
//!     worker_id: "worker-123".to_string(),
//!     model_id: "meta-llama/Llama-3-8b".to_string(),
//!     device: "cuda:0".to_string(),
//!     port: 9001,
//!     queen_url: "http://localhost:8500".to_string(),
//!     job_id: "job-456".to_string(),
//! };
//!
//! let result = spawn_worker(config).await?;
//! println!("Worker spawned: PID {}", result.pid);
//! # Ok(())
//! # }
//! ```

// TEAM-271: Module declarations
// TEAM-276: Renamed modules for consistency across lifecycle crates
/// Worker start operations (renamed from spawn)
pub mod start;
/// Worker stop operations (renamed from delete)
pub mod stop;
/// List worker processes (renamed from process_list)
pub mod list;
/// Get worker process details (renamed from process_get)
pub mod get;
/// Install worker binary (stub for consistency)
pub mod install;
/// Uninstall worker binary (stub for consistency)
pub mod uninstall;
/// Request/Response types
pub mod types;

// NOTE: WorkerList and WorkerGet are NOT implemented in hive
// According to corrected architecture (CORRECTION_269_TO_272_ARCHITECTURE_FIX.md):
// - Hive is STATELESS executor
// - Worker tracking happens in QUEEN via heartbeats
// - WorkerList/WorkerGet should query queen's registry, not hive
// - WorkerDelete is the only operation that makes sense in hive (kill process by PID)
//
// TEAM-274: WorkerProcessList/Get/Delete are hive-local operations
// - These use local `ps` commands to inspect processes on this hive
// - This is different from ActiveWorkerList (queen's registry via heartbeats)
// - WorkerProcessXxx = local process management, ActiveWorkerXxx = distributed tracking

// TEAM-271: Re-export main types and functions
// TEAM-276: Updated exports for renamed modules and added install/uninstall stubs
pub use start::start_worker;
pub use stop::stop_worker;
pub use list::list_workers;
pub use get::get_worker;
pub use install::install_worker;
pub use uninstall::uninstall_worker;
pub use types::{StartResult, WorkerStartConfig, WorkerInfo};
