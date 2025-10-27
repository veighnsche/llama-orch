//! Uninstall rbee-hive from remote host
//!
//! TEAM-290: Remote hive uninstallation via SSH

use anyhow::{Context, Result};
use observability_narration_core::n;
use ssh_config::SshClient; // TEAM-314: Use shared SSH client

// TEAM-314: All narration migrated to n!() macro

/// Uninstall rbee-hive from remote host
///
/// # Arguments
/// * `host` - SSH host alias
/// * `install_dir` - Remote installation directory
pub async fn uninstall_hive(host: &str, install_dir: &str) -> Result<()> {
    n!("uninstall_hive_start", "🗑️  Uninstalling rbee-hive from '{}'", host);

    let client = SshClient::connect(host).await?;
    let remote_path = format!("{}/rbee-hive", install_dir);

    // Check if hive is running
    let is_running = client
        .execute("pgrep -f rbee-hive")
        .await
        .is_ok();

    if is_running {
        n!("uninstall_hive_stop", "⚠️  Stopping rbee-hive on '{}'", host);

        // Stop hive
        client
            .execute("pkill -f rbee-hive")
            .await
            .ok(); // Ignore errors
    }

    // Remove binary
    client
        .execute(&format!("rm -f {}", remote_path))
        .await
        .context("Failed to remove hive binary")?;

    n!("uninstall_hive_complete", "✅ Hive uninstalled from '{}'", host);

    Ok(())
}
