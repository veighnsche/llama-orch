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
//! # Module Structure
//!
//! - `types` - Request/Response types for all operations
//! - `spawn` - Worker spawning operations (TEAM-271)
//! - `list` - List workers (TEAM-272) - queries queen
//! - `get` - Get worker details (TEAM-272) - queries queen
//! - `delete` - Delete worker (TEAM-272) - kills process
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
/// Request/Response types
pub mod types;
/// Worker spawning operations
pub mod spawn;

// TEAM-272: Worker management operations
/// Worker deletion (process cleanup)
pub mod delete;

// TEAM-274: Worker process operations (hive-local, stateless)
/// List worker processes using local ps
pub mod process_list;
/// Get worker process details by PID
pub mod process_get;

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
pub use spawn::spawn_worker;
pub use types::{SpawnResult, WorkerSpawnConfig};

// TEAM-272: Re-export worker deletion
pub use delete::delete_worker;

// TEAM-274: Re-export process operations
pub use process_list::{list_worker_processes, WorkerProcessInfo};
pub use process_get::get_worker_process;
