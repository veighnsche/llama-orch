//! Stop rbee-hive on local or remote host
//!
//! TEAM-290: Local or remote hive shutdown
//! TEAM-291: Added graceful shutdown (SIGTERM → wait → SIGKILL)

use anyhow::{Context, Result};
use observability_narration_core::NarrationFactory;

use crate::ssh::SshClient;

const NARRATE: NarrationFactory = NarrationFactory::new("hive-stop");

/// Stop rbee-hive on local or remote host
///
/// # Arguments
/// * `host` - Host to stop on ("localhost" for local, SSH alias for remote)
pub async fn stop_hive(host: &str) -> Result<()> {
    NARRATE
        .action("stop_hive")
        .context(host)
        .human("⏹️  Stopping rbee-hive on '{}'")
        .emit();

    // Check if localhost (direct stop) or remote (SSH stop)
    if host == "localhost" || host == "127.0.0.1" {
        stop_hive_local().await
    } else {
        stop_hive_remote(host).await
    }
}

/// Stop rbee-hive locally (no SSH)
/// 
/// TEAM-291: Graceful shutdown pattern (SIGTERM → wait → SIGKILL)
async fn stop_hive_local() -> Result<()> {
    NARRATE
        .action("stop_hive_local")
        .human("⏹️  Stopping rbee-hive locally...")
        .emit();

    // ============================================================
    // TEAM-291: Graceful shutdown pattern
    // ============================================================
    // PATTERN: SIGTERM (graceful) → wait 5s → SIGKILL (force)
    // This matches daemon-lifecycle pattern used in queen/keeper
    // ============================================================

    // Step 1: Try graceful shutdown (SIGTERM)
    NARRATE
        .action("stop_hive_sigterm")
        .human("📨 Sending SIGTERM (graceful shutdown)...")
        .emit();

    let output = tokio::process::Command::new("pkill")
        .arg("-TERM") // TEAM-291: Explicit SIGTERM for graceful shutdown
        .arg("-f")
        .arg("rbee-hive")
        .output()
        .await
        .context("Failed to execute pkill")?;

    if !output.status.success() {
        // pkill returns non-zero if no process found, which is fine
        NARRATE
            .action("stop_hive_not_running")
            .human("ℹ️  rbee-hive was not running")
            .emit();
        return Ok(());
    }

    // Step 2: Wait for graceful shutdown (5 seconds)
    NARRATE
        .action("stop_hive_wait")
        .human("⏳ Waiting for graceful shutdown (5 seconds)...")
        .emit();

    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;

    // Step 3: Check if still running
    let output = tokio::process::Command::new("pgrep")
        .arg("-f")
        .arg("rbee-hive")
        .output()
        .await
        .context("Failed to execute pgrep")?;

    if output.status.success() {
        // Still running - force kill
        NARRATE
            .action("stop_hive_sigkill")
            .human("⚠️  Graceful shutdown failed, sending SIGKILL (force)...")
            .emit();

        let output = tokio::process::Command::new("pkill")
            .arg("-KILL") // TEAM-291: Force kill
            .arg("-f")
            .arg("rbee-hive")
            .output()
            .await
            .context("Failed to execute pkill -KILL")?;

        if !output.status.success() {
            anyhow::bail!("Failed to force kill hive (pkill -KILL failed)");
        }

        // Wait a bit for force kill
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

        // Final verification
        let output = tokio::process::Command::new("pgrep")
            .arg("-f")
            .arg("rbee-hive")
            .output()
            .await
            .context("Failed to execute pgrep")?;

        if output.status.success() {
            anyhow::bail!("Hive failed to stop even after SIGKILL (still running)");
        }

        NARRATE
            .action("stop_hive_force_complete")
            .human("✅ Hive stopped (force killed)")
            .emit();
    } else {
        NARRATE
            .action("stop_hive_graceful_complete")
            .human("✅ Hive stopped (graceful shutdown)")
            .emit();
    }

    Ok(())
}

/// Stop rbee-hive remotely via SSH
async fn stop_hive_remote(host: &str) -> Result<()> {
    NARRATE
        .action("stop_hive_remote")
        .context(host)
        .human("⏹️  Stopping rbee-hive on '{}' via SSH...")
        .emit();

    let client = SshClient::connect(host).await?;

    // Stop hive
    client
        .execute("pkill -f rbee-hive")
        .await
        .context("Failed to stop hive")?;

    // Wait a bit
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // Verify it's stopped
    let is_running = client
        .execute("pgrep -f rbee-hive")
        .await
        .is_ok();

    if is_running {
        anyhow::bail!("Hive failed to stop on '{}'", host);
    }

    NARRATE
        .action("stop_hive_complete")
        .context(host)
        .human("✅ Hive stopped on '{}'")
        .emit();

    Ok(())
}
