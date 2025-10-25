//! Queen uninstall operation
//!
//! TEAM-276: Extracted from rbee-keeper/src/handlers/queen.rs
//! TEAM-262: Uninstall queen binary
//! TEAM-263: Implemented uninstall logic
//! TEAM-296: Enhanced to check if installed first (delegates to daemon-lifecycle)

use anyhow::Result;
use daemon_lifecycle::{uninstall_daemon, UninstallConfig};
use observability_narration_core::NarrationFactory;

const NARRATE: NarrationFactory = NarrationFactory::new("queen-life");

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
/// * `queen_url` - Base URL for queen (e.g., "http://localhost:8500") for health check
///
/// # Returns
/// * `Ok(())` - Uninstallation successful
/// * `Err` - Not installed, queen is running, or removal failed
pub async fn uninstall_queen(queen_url: &str) -> Result<()> {
    NARRATE.action("queen_uninstall").human("🗑️ Uninstalling queen-rbee...").emit();

    // Determine install location
    let home = std::env::var("HOME")?;
    let install_path = std::path::PathBuf::from(format!("{}/.local/bin/queen-rbee", home));

    // TEAM-296: Check if installed first
    if !install_path.exists() {
        NARRATE
            .action("queen_uninstall")
            .human("❌ Queen not installed")
            .error_kind("not_installed")
            .emit();
        anyhow::bail!("Queen not installed. Nothing to uninstall.");
    }

    // Use daemon-lifecycle to handle uninstallation (checks if running + removes binary)
    let config = UninstallConfig {
        daemon_name: "queen-rbee".to_string(),
        install_path: install_path.clone(),
        health_url: Some(queen_url.to_string()),
        health_timeout_secs: Some(2),
        job_id: None,
    };

    uninstall_daemon(config).await?;

    NARRATE.action("queen_uninstall").human("✅ Queen uninstalled successfully!").emit();
    Ok(())
}
