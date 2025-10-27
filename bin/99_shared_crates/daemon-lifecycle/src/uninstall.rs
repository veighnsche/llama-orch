//! Daemon uninstallation module
//!
//! TEAM-312: Extracted from install.rs to separate module

use anyhow::Result;
use observability_narration_core::n;
use observability_narration_macros::with_job_id;

use crate::install::UninstallConfig;

/// Uninstall a daemon binary
///
/// Steps:
/// 1. Check if binary exists at install_path
/// 2. If health_url provided, check if daemon is running (error if yes)
/// 3. Remove binary file
/// 4. Emit success narration
///
/// # Arguments
/// * `config` - Uninstallation configuration
///
/// # Errors
/// This function will return an error if:
/// * The daemon is currently running (must be stopped first)
/// * Failed to remove the binary file from the filesystem
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::{UninstallConfig, uninstall_daemon};
/// use std::path::PathBuf;
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = UninstallConfig {
///     daemon_name: "queen-rbee".to_string(),
///     install_path: PathBuf::from("/home/user/.local/bin/queen-rbee"),
///     health_url: Some("http://localhost:8500".to_string()),
///     health_timeout_secs: Some(2),
///     job_id: None,
/// };
///
/// uninstall_daemon(config).await?;
/// # Ok(())
/// # }
/// ```
#[with_job_id] // TEAM-328: Eliminates job_id context boilerplate
pub async fn uninstall_daemon(config: UninstallConfig) -> Result<()> {
    n!("daemon_uninstall", "🗑️  Uninstalling daemon '{}'", config.daemon_name);

    // Step 1: Check if binary exists
    let install_path = std::path::Path::new(&config.install_path);
    if !install_path.exists() {
        n!("daemon_not_installed", "⚠️  Daemon '{}' not installed at: {}", config.daemon_name, install_path.display());
        return Ok(());
    }

    // Step 2: Check if daemon is running (if health_url provided)
    if let Some(health_url) = config.health_url {
        let timeout_secs = config.health_timeout_secs.unwrap_or(2);
        let is_running = crate::health::is_daemon_healthy(
            &health_url,
            None, // Use default /health endpoint
            Some(std::time::Duration::from_secs(timeout_secs)),
        )
        .await;

        if is_running {
            n!("daemon_still_running", "⚠️  Daemon '{}' is currently running. Stop it first.", config.daemon_name);
            anyhow::bail!("Daemon {} is still running", config.daemon_name);
        }
    }

    // Step 3: Remove binary file
    std::fs::remove_file(&config.install_path)?;

    n!("daemon_uninstalled", "✅ Daemon '{}' uninstalled successfully!", config.daemon_name);
    n!("daemon_removed", "🗑️  Removed: {}", install_path.display());

    Ok(())
}
