//! Command handlers for rbee-keeper
//!
//! TEAM-276: Extracted from main.rs for better organization
//! TEAM-282: Added package manager handlers (sync, package_status, validate, migrate)
//! TEAM-284: DELETED package manager handlers (SSH/remote operations removed)
//! TEAM-313: Remove hive_check module (moving to rbee-hive)
//! TEAM-380: Split hive into hive_lifecycle and hive_jobs
//!
//! Each handler module implements the business logic for a specific
//! command category (queen, hive, worker, model, infer).

// TEAM-380: Split hive into lifecycle and jobs
// TEAM-324: Made public so HiveLifecycleAction can be re-exported from cli/mod.rs
pub mod hive_lifecycle;
// TEAM-380: New module for hive job operations (uses job-client)
// Contains both HiveJobsAction enum and handler functions
pub mod hive_jobs;
mod infer;
// TEAM-324: Made public so ModelAction can be re-exported from cli/mod.rs
pub mod model;
// TEAM-324: Made public so QueenAction can be re-exported from cli/mod.rs
pub mod queen;
mod self_check;
mod status;
// TEAM-324: Made public so WorkerAction can be re-exported from cli/mod.rs
pub mod worker;
// TEAM-284: DELETED migrate, package_status, sync, validate handlers

// TEAM-380: Export hive lifecycle handler
pub use hive_lifecycle::handle_hive_lifecycle;
// TEAM-380: Export hive jobs handler and helpers
pub use hive_jobs::{handle_hive_jobs, submit_hive_job, get_hive_url};
pub use infer::handle_infer;
pub use model::handle_model;
pub use queen::handle_queen;
pub use self_check::handle_self_check;
pub use status::handle_status;
pub use worker::handle_worker;
// TEAM-284: DELETED handle_migrate, handle_package_status, handle_sync, handle_validate
