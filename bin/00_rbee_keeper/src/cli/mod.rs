//! CLI argument definitions for rbee-keeper
//!
//! TEAM-276: Extracted from main.rs for better organization

mod commands;
mod hive;
mod model;
mod queen;
mod worker;

pub use commands::{Cli, Commands};
pub use hive::HiveAction;
pub use model::ModelAction;
pub use queen::QueenAction;
// TEAM-278: DELETED WorkerBinaryAction - no longer exists
pub use worker::{WorkerAction, WorkerProcessAction};
