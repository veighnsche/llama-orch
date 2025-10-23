//! Queen-rbee command handlers
//!
//! TEAM-276: Extracted from main.rs
//! TEAM-276: Refactored to delegate to queen-lifecycle crate
//!
//! This module is now a thin wrapper that delegates all queen lifecycle
//! operations to the queen-lifecycle crate. All business logic lives there.

use anyhow::Result;
use queen_lifecycle::{
    check_queen_status, get_queen_info, install_queen, rebuild_queen, start_queen, stop_queen,
    uninstall_queen,
};

use crate::cli::QueenAction;

/// Handle queen-rbee lifecycle commands
///
/// Delegates to queen-lifecycle crate for all operations.
pub async fn handle_queen(action: QueenAction, queen_url: &str) -> Result<()> {
    match action {
        QueenAction::Start => start_queen(queen_url).await,
        QueenAction::Stop => stop_queen(queen_url).await,
        QueenAction::Status => check_queen_status(queen_url).await,
        QueenAction::Rebuild { with_local_hive } => rebuild_queen(with_local_hive).await,
        QueenAction::Info => get_queen_info(queen_url).await,
        QueenAction::Install { binary } => install_queen(binary).await,
        QueenAction::Uninstall => uninstall_queen(queen_url).await,
    }
}
