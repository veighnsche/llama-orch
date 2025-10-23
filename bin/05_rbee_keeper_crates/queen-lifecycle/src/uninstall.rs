//! Queen uninstall operation
//!
//! TEAM-276: Extracted from rbee-keeper/src/handlers/queen.rs
//! TEAM-262: Uninstall queen binary
//! TEAM-263: Implemented uninstall logic

use anyhow::Result;
use daemon_lifecycle::{uninstall_daemon, UninstallConfig};

/// Uninstall queen-rbee binary from ~/.local/bin
///
/// Steps:
/// 1. Check if binary exists at ~/.local/bin/queen-rbee
/// 2. Check if queen is running (error if yes)
/// 3. Remove binary file
///
/// # Arguments
/// * `queen_url` - Base URL for queen (e.g., "http://localhost:8500") for health check
///
/// # Returns
/// * `Ok(())` - Uninstallation successful (or binary not found)
/// * `Err` - Queen is running or removal failed
pub async fn uninstall_queen(queen_url: &str) -> Result<()> {
    // Determine install location
    let home = std::env::var("HOME")?;
    let install_path = std::path::PathBuf::from(format!("{}/.local/bin/queen-rbee", home));

    // Use daemon-lifecycle to handle uninstallation
    let config = UninstallConfig {
        daemon_name: "queen-rbee".to_string(),
        install_path,
        health_url: Some(queen_url.to_string()),
        health_timeout_secs: Some(2),
        job_id: None,
    };

    uninstall_daemon(config).await
}
