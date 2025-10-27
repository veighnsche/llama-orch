//! Command handlers for rbee-keeper
//!
//! TEAM-276: Extracted from main.rs for better organization
//! TEAM-282: Added package manager handlers (sync, package_status, validate, migrate)
//! TEAM-284: DELETED package manager handlers (SSH/remote operations removed)
//! TEAM-313: Remove hive_check module (moving to rbee-hive)
//!
//! Each handler module implements the business logic for a specific
//! command category (queen, hive, worker, model, infer).

// TEAM-324: Made public so HiveAction can be re-exported from cli/mod.rs
pub mod hive;
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

pub use hive::handle_hive;
pub use infer::handle_infer;
pub use model::handle_model;
pub use queen::handle_queen;
pub use self_check::handle_self_check;
pub use status::handle_status;
pub use worker::handle_worker;
// TEAM-284: DELETED handle_migrate, handle_package_status, handle_sync, handle_validate
