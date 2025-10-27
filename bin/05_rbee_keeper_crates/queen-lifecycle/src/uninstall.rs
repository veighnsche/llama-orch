//! Queen uninstall operation
//!
//! TEAM-276: Extracted from rbee-keeper/src/handlers/queen.rs
//! TEAM-262: Uninstall queen binary
//! TEAM-263: Implemented uninstall logic
//! TEAM-296: Enhanced to check if installed first (delegates to daemon-lifecycle)
//! TEAM-312: Migrated to n!() macro

use anyhow::Result;
use daemon_lifecycle::{uninstall_daemon, UninstallConfig};
use observability_narration_core::n;

/// Uninstall queen-rbee binary from ~/.local/bin
///
/// TEAM-296: Enhanced to check if installed first
///
/// Steps:
/// 1. Check if binary exists at ~/.local/bin/queen-rbee (error if not)
/// 2. Check if queen is running (error if yes)
/// 3. Remove binary file (via daemon-lifecycle)
///
/// # Arguments
/// * `queen_url` - Base URL for queen (e.g., "http://localhost:7833") for health check
///
/// # Errors
/// This function will return an error if:
/// * HOME environment variable is not set
/// * Queen is not installed at ~/.local/bin/queen-rbee
/// * Queen is currently running (must be stopped first)
/// * Failed to remove the binary file
pub async fn uninstall_queen(queen_url: &str) -> Result<()> {
    n!("start", "🗑️ Uninstalling queen-rbee...");

    // Determine install location
    let home = std::env::var("HOME")?;
    let install_path = std::path::PathBuf::from(format!("{}/.local/bin/queen-rbee", home));

    // TEAM-296: Check if installed first
    if !install_path.exists() {
        n!("not_installed", "❌ Queen not installed");
        anyhow::bail!("Queen not installed. Nothing to uninstall.");
    }

    // Use daemon-lifecycle to handle uninstallation (checks if running + removes binary)
    let config = UninstallConfig {
        daemon_name: "queen-rbee".to_string(),
        install_path: install_path.display().to_string(),
        health_url: Some(queen_url.to_string()),
        health_timeout_secs: Some(2),
        job_id: None,
    };

    uninstall_daemon(config).await?;

    n!("done", "✅ Queen uninstalled successfully!");
    Ok(())
}
