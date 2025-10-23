//! Command handlers for rbee-keeper
//!
//! TEAM-276: Extracted from main.rs for better organization
//! TEAM-282: Added package manager handlers (sync, package_status, validate, migrate)
//!
//! Each handler module implements the business logic for a specific
//! command category (queen, hive, worker, model, infer).

mod hive;
mod infer;
mod migrate;
mod model;
mod package_status;
mod queen;
mod status;
mod sync;
mod validate;
mod worker;

pub use hive::handle_hive;
pub use infer::handle_infer;
pub use migrate::handle_migrate;
pub use model::handle_model;
pub use package_status::handle_package_status;
pub use queen::handle_queen;
pub use status::handle_status;
pub use sync::handle_sync;
pub use validate::handle_validate;
pub use worker::handle_worker;
