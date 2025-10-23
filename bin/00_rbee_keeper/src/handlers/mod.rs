//! Command handlers for rbee-keeper
//!
//! TEAM-276: Extracted from main.rs for better organization
//!
//! Each handler module implements the business logic for a specific
//! command category (queen, hive, worker, model, infer).

mod hive;
mod infer;
mod model;
mod queen;
mod status;
mod worker;

pub use hive::handle_hive;
pub use infer::handle_infer;
pub use model::handle_model;
pub use queen::handle_queen;
pub use status::handle_status;
pub use worker::handle_worker;
