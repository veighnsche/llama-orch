//! Queen start operation
//!
//! TEAM-276: Extracted from rbee-keeper/src/handlers/queen.rs

use anyhow::Result;
use observability_narration_core::NarrationFactory;

use crate::ensure::ensure_queen_running;

const NARRATE: NarrationFactory = NarrationFactory::new("queen-life");

/// Start queen-rbee daemon
///
/// Uses the ensure_queen_running pattern to start queen if not already running.
///
/// # Arguments
/// * `queen_url` - Base URL for queen (e.g., "http://localhost:8500")
///
/// # Returns
/// * `Ok(())` - Queen started successfully
///
/// # Errors
/// * Returns error if queen binary not found
/// * Returns error if queen fails to spawn
/// * Returns error if queen fails health check
pub async fn start_queen(queen_url: &str) -> Result<()> {
    let queen_handle = ensure_queen_running(queen_url).await?;

    NARRATE
        .action("queen_start")
        .context(queen_handle.base_url())
        .human("✅ Queen started on {}")
        .emit();

    // Keep queen alive - detach the handle
    // The queen process will continue running independently
    drop(queen_handle);

    Ok(())
}
