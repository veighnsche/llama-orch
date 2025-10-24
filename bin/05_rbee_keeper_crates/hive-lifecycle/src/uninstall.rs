//! Uninstall rbee-hive from remote host
//!
//! TEAM-290: Remote hive uninstallation via SSH

use anyhow::{Context, Result};
use observability_narration_core::NarrationFactory;

use crate::ssh::SshClient;

const NARRATE: NarrationFactory = NarrationFactory::new("hive-rm");

/// Uninstall rbee-hive from remote host
///
/// # Arguments
/// * `host` - SSH host alias
/// * `install_dir` - Remote installation directory
pub async fn uninstall_hive(host: &str, install_dir: &str) -> Result<()> {
    NARRATE
        .action("uninstall_hive_start")
        .context(host)
        .human("🗑️  Uninstalling rbee-hive from '{}'")
        .emit();

    let client = SshClient::connect(host).await?;
    let remote_path = format!("{}/rbee-hive", install_dir);

    // Check if hive is running
    let is_running = client
        .execute("pgrep -f rbee-hive")
        .await
        .is_ok();

    if is_running {
        NARRATE
            .action("uninstall_hive_stop")
            .context(host)
            .human("⚠️  Stopping rbee-hive on '{}'")
            .emit();

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

    NARRATE
        .action("uninstall_hive_complete")
        .context(host)
        .human("✅ Hive uninstalled from '{}'")
        .emit();

    Ok(())
}
