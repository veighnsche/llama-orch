//! Queen lifecycle types
//!
//! TEAM-259: Extracted from rbee-keeper/src/queen_lifecycle.rs

use anyhow::Result;
use observability_narration_core::NarrationFactory;

// TEAM-192: Local narration factory for queen lifecycle
const NARRATE: NarrationFactory = NarrationFactory::new("kpr-life");

/// Handle to the queen-rbee process
///
/// Tracks whether rbee-keeper started the queen and provides cleanup.
/// IMPORTANT: Only shuts down queen if rbee-keeper started it!
pub struct QueenHandle {
    /// True if rbee-keeper started the queen (must cleanup)
    /// False if queen was already running (don't touch it)
    started_by_us: bool,

    /// Base URL of the queen
    base_url: String,

    /// Process ID if we started it
    pid: Option<u32>,
}

impl QueenHandle {
    /// Create handle for queen that was already running
    pub fn already_running(base_url: String) -> Self {
        Self { started_by_us: false, base_url, pid: None }
    }

    /// Create handle for queen that we just started
    pub fn started_by_us(base_url: String, pid: Option<u32>) -> Self {
        Self { started_by_us: true, base_url, pid }
    }

    /// Check if we started the queen (and should clean it up)
    pub fn should_cleanup(&self) -> bool {
        self.started_by_us
    }

    /// Get the queen's base URL
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Keep the queen alive (no shutdown after task)
    ///
    /// Queen stays running for future tasks. After 5 minutes of inactivity,
    /// the hive will automatically purge workers (handled by rbee-hive).
    ///
    /// # Returns
    /// * `Ok(())` - Always succeeds (queen stays alive)
    ///
    /// # Errors
    ///
    /// Currently never returns an error
    pub async fn shutdown(self) -> Result<()> {
        NARRATE
            .action("queen_stop")
            .human("Task complete, keeping queen alive for future tasks")
            .emit();
        Ok(())
    }
}
