//! CLI argument definitions for rbee-keeper
//!
//! TEAM-276: Extracted from main.rs for better organization
//! TEAM-324: All action enums moved to handlers to eliminate duplication

mod commands;

pub use commands::{Cli, Commands};
// TEAM-324: Re-export from handlers (single source of truth)
pub use crate::handlers::hive::HiveAction;
pub use crate::handlers::model::ModelAction;
pub use crate::handlers::queen::QueenAction;
pub use crate::handlers::worker::{WorkerAction, WorkerProcessAction};
