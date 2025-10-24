//! Start rbee-hive on local or remote host
//!
//! TEAM-290: Local or remote hive startup
//! TEAM-291: Added comprehensive error handling and crash detection

use anyhow::{Context, Result};
use daemon_lifecycle::DaemonManager;
use observability_narration_core::NarrationFactory;

use crate::ssh::SshClient;

const NARRATE: NarrationFactory = NarrationFactory::new("hive-strt");

/// Start rbee-hive on local or remote host
///
/// TEAM-290: Supports both local and remote startup
/// TEAM-292: Added queen_url parameter for automatic queen discovery
///
/// # Arguments
/// * `host` - Host to start on ("localhost" for local, SSH alias for remote)
/// * `install_dir` - Installation directory
/// * `port` - Hive HTTP port (default: 9000)
/// * `queen_url` - Optional queen URL (if None, uses default http://localhost:8500)
pub async fn start_hive(host: &str, install_dir: &str, port: u16, queen_url: Option<&str>) -> Result<()> {
    NARRATE
        .action("start_hive")
        .context(host)
        .human("▶️  Starting rbee-hive on '{}'")
        .emit();

    // TEAM-292: Use provided queen_url or default
    let queen_url = queen_url.unwrap_or("http://localhost:8500");

    // Check if localhost (direct start) or remote (SSH start)
    if host == "localhost" || host == "127.0.0.1" {
        start_hive_local(install_dir, port, queen_url).await
    } else {
        start_hive_remote(host, install_dir, port, queen_url).await
    }
}

/// Start rbee-hive locally (no SSH)
///
/// TEAM-292: Added queen_url parameter
async fn start_hive_local(install_dir: &str, port: u16, queen_url: &str) -> Result<()> {
    NARRATE
        .action("start_hive_local")
        .context(queen_url)
        .human("▶️  Starting rbee-hive locally (queen: {})...")
        .emit();

    // Find binary
    let binary_path = if install_dir.contains(".local/bin") {
        // Installed location
        std::path::PathBuf::from(install_dir).join("rbee-hive")
    } else {
        // Try to find in target directory
        DaemonManager::find_in_target("rbee-hive")?
    };

    if !binary_path.exists() {
        anyhow::bail!("rbee-hive binary not found at: {}", binary_path.display());
    }

    // ============================================================
    // TEAM-291: Enhanced startup with crash detection
    // ============================================================
    // PROBLEM: Old code used Stdio::null() which hid startup errors
    // If hive crashed immediately (e.g., routing panic), we never saw why
    //
    // SOLUTION:
    // 1. Capture stderr to temp file for crash diagnostics
    // 2. Spawn process and get PID
    // 3. Wait for startup (2 seconds)
    // 4. Check if process still running (detect crashes)
    // 5. Verify HTTP server responds (detect silent failures)
    // 6. Show stderr if anything went wrong
    // ============================================================

    // Create temp file for stderr capture
    let stderr_path = format!("/tmp/rbee-hive-{}.stderr", std::process::id());
    let stderr_file = std::fs::File::create(&stderr_path)
        .context("Failed to create stderr capture file")?;

    NARRATE
        .action("start_hive_spawn")
        .context(binary_path.display().to_string())
        .human("🚀 Spawning hive from '{}'")
        .emit();

    // Start hive in background with stderr capture
    // TEAM-292: Pass queen URL and hive ID to the hive
    let mut child = tokio::process::Command::new(&binary_path)
        .arg("--port")
        .arg(port.to_string())
        .arg("--queen-url")
        .arg(queen_url)
        .arg("--hive-id")
        .arg("localhost") // TEAM-292: Default hive ID for local hives
        .stdout(std::process::Stdio::null())
        .stderr(stderr_file) // TEAM-291: Capture stderr for diagnostics
        .spawn()
        .context("Failed to spawn rbee-hive")?;

    let pid = child.id().ok_or_else(|| anyhow::anyhow!("Failed to get PID"))?;

    NARRATE
        .action("start_hive_spawned")
        .context(pid.to_string())
        .human("✅ Hive spawned with PID: {}")
        .emit();

    // TEAM-291: Wait for startup and check for crashes
    NARRATE
        .action("start_hive_wait")
        .human("⏳ Waiting for hive to start (2 seconds)...")
        .emit();

    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // TEAM-291: Check if process crashed during startup
    match child.try_wait() {
        Ok(Some(status)) => {
            // Process exited - this is a crash!
            let stderr_content = std::fs::read_to_string(&stderr_path)
                .unwrap_or_else(|_| "(failed to read stderr)".to_string());
            
            NARRATE
                .action("start_hive_crashed")
                .context(status.to_string())
                .human("❌ Hive crashed during startup: {}")
                .emit();

            anyhow::bail!(
                "Hive crashed during startup (exit code: {})\n\nStderr:\n{}",
                status,
                stderr_content
            );
        }
        Ok(None) => {
            // Process still running - good!
            NARRATE
                .action("start_hive_alive")
                .context(pid.to_string())
                .human("✅ Process still alive (PID: {})")
                .emit();
        }
        Err(e) => {
            anyhow::bail!("Failed to check process status: {}", e);
        }
    }

    // TEAM-291: Verify HTTP server is responding
    NARRATE
        .action("start_hive_health_check")
        .context(format!("http://localhost:{}/health", port))
        .human("🏥 Checking health endpoint: {}")
        .emit();

    let health_url = format!("http://localhost:{}/health", port);
    let client = reqwest::Client::new();
    
    match client.get(&health_url).timeout(std::time::Duration::from_secs(3)).send().await {
        Ok(response) if response.status().is_success() => {
            NARRATE
                .action("start_hive_health_ok")
                .human("✅ Health check passed")
                .emit();
        }
        Ok(response) => {
            let stderr_content = std::fs::read_to_string(&stderr_path)
                .unwrap_or_else(|_| "(failed to read stderr)".to_string());
            
            anyhow::bail!(
                "Hive HTTP server returned error status: {}\n\nStderr:\n{}",
                response.status(),
                stderr_content
            );
        }
        Err(e) => {
            let stderr_content = std::fs::read_to_string(&stderr_path)
                .unwrap_or_else(|_| "(failed to read stderr)".to_string());
            
            anyhow::bail!(
                "Hive HTTP server not responding: {}\n\nStderr:\n{}",
                e,
                stderr_content
            );
        }
    }

    // TEAM-291: Clean up stderr file on success
    let _ = std::fs::remove_file(&stderr_path);

    NARRATE
        .action("start_hive_complete")
        .context(format!("http://localhost:{}", port))
        .human("✅ Hive started at '{}'")
        .emit();

    Ok(())
}

/// Start rbee-hive remotely via SSH
///
/// TEAM-292: Added queen_url parameter
async fn start_hive_remote(host: &str, install_dir: &str, port: u16, queen_url: &str) -> Result<()> {
    NARRATE
        .action("start_hive_remote")
        .context(host)
        .context(queen_url)
        .human("▶️  Starting rbee-hive on '{}' via SSH (queen: {})...")
        .emit();

    let client = SshClient::connect(host).await?;
    let remote_path = format!("{}/rbee-hive", install_dir);

    // Start hive in background (nohup pattern from daemon-lifecycle)
    // TEAM-292: Pass queen URL and hive ID to remote hive
    client
        .execute(&format!(
            "nohup {} --port {} --queen-url {} --hive-id {} > /dev/null 2>&1 &",
            remote_path, port, queen_url, host
        ))
        .await
        .context("Failed to start hive")?;

    // Wait a bit for startup
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // Verify it's running
    let is_running = client
        .execute("pgrep -f rbee-hive")
        .await
        .is_ok();

    if !is_running {
        anyhow::bail!("Hive failed to start on '{}'", host);
    }

    NARRATE
        .action("start_hive_complete")
        .context(host)
        .human("✅ Hive started on '{}'")
        .emit();

    Ok(())
}
