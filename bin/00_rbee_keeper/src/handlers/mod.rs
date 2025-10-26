//! Command handlers for rbee-keeper
//!
//! TEAM-276: Extracted from main.rs for better organization
//! TEAM-282: Added package manager handlers (sync, package_status, validate, migrate)
//! TEAM-284: DELETED package manager handlers (SSH/remote operations removed)
//!
//! Each handler module implements the business logic for a specific
//! command category (queen, hive, worker, model, infer).

mod hive;
mod infer;
mod model;
mod queen;
mod self_check;
mod status;
mod worker;
// TEAM-284: DELETED migrate, package_status, sync, validate handlers

pub use hive::handle_hive;
pub use infer::handle_infer;
pub use model::handle_model;
pub use queen::handle_queen;
pub use self_check::handle_self_check;
pub use status::handle_status;
pub use worker::handle_worker;
// TEAM-284: DELETED handle_migrate, handle_package_status, handle_sync, handle_validate
