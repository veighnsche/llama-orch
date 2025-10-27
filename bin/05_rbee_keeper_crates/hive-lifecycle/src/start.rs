//! Start rbee-hive on local or remote host
//!
//! TEAM-290: Local or remote hive startup
//! TEAM-291: Added comprehensive error handling and crash detection

use anyhow::{Context, Result};
use daemon_lifecycle::DaemonManager;
use observability_narration_core::n;
use ssh_config::SshClient; // TEAM-314: Use shared SSH client

use crate::DEFAULT_INSTALL_DIR;

// TEAM-314: All narration migrated to n!() macro

/// Start rbee-hive on local or remote host
///
/// TEAM-290: Supports both local and remote startup
/// TEAM-292: Added queen_url parameter for automatic queen discovery
///
/// # Arguments
/// * `host` - Host to start on ("localhost" for local, SSH alias for remote)
/// * `install_dir` - Installation directory
/// * `port` - Hive HTTP port (default: 7835)
/// * `queen_url` - Optional queen URL (if None, uses default http://localhost:7833)
pub async fn start_hive(host: &str, install_dir: &str, port: u16, queen_url: Option<&str>) -> Result<()> {
    n!("start_hive", "▶️  Starting rbee-hive on '{}'", host);

    // Check if localhost (direct start) or remote (SSH start)
    if host == "localhost" || host == "127.0.0.1" {
        // TEAM-292: Use provided queen_url or default for localhost
        let queen_url = queen_url.unwrap_or("http://localhost:7833");
        start_hive_local(install_dir, port, queen_url).await
    } else {
        // TEAM-314: Pass queen_url as Option for remote hives (may be None)
        start_hive_remote(host, install_dir, port, queen_url).await
    }
}

/// Start rbee-hive locally (no SSH)
///
/// TEAM-292: Added queen_url parameter
/// TEAM-314: Added detailed narration and error handling
async fn start_hive_local(install_dir: &str, port: u16, queen_url: &str) -> Result<()> {
    n!("start_hive_local", "▶️  Starting rbee-hive locally (queen: {})...", queen_url);

    // Find binary
    n!("hive_binary_resolve", "🔍 Resolving hive binary location...");
    let binary_path = if install_dir.contains(".local/bin") {
        // Installed location
        let path = std::path::PathBuf::from(install_dir).join("rbee-hive");
        n!("hive_binary_check", "Checking installed location: {}", path.display());
        path
    } else {
        // Try to find in target directory
        n!("hive_binary_check", "Checking development build in target/...");
        DaemonManager::find_in_target("rbee-hive")?
    };

    if !binary_path.exists() {
        n!("hive_binary_missing", "❌ Hive binary not found at: {}", binary_path.display());
        anyhow::bail!(
            "rbee-hive binary not found at: {}\nRun 'rbee hive install' to install from source.",
            binary_path.display()
        );
    }
    
    n!("hive_binary_found", "✅ Hive binary found at: {}", binary_path.display());

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
    n!("hive_stderr_setup", "📄 Setting up stderr capture at {}", stderr_path);
    let stderr_file = std::fs::File::create(&stderr_path)
        .context("Failed to create stderr capture file")?;

    n!("start_hive_spawn", "🚀 Spawning hive from '{}'", binary_path.display());

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

    n!("start_hive_spawned", "✅ Hive spawned with PID: {}", pid);

    // TEAM-291: Wait for startup and check for crashes
    n!("start_hive_wait", "⏳ Waiting for hive to start (2 seconds)...");

    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // TEAM-291: Check if process crashed during startup
    n!("hive_process_check", "🔍 Checking if hive process is still alive...");
    match child.try_wait() {
        Ok(Some(status)) => {
            // Process exited - this is a crash!
            let stderr_content = std::fs::read_to_string(&stderr_path)
                .unwrap_or_else(|_| "(failed to read stderr)".to_string());
            
            n!("start_hive_crashed", "❌ Hive crashed during startup: {}", status);
            n!("hive_crash_stderr", "Stderr output:\n{}", stderr_content);

            anyhow::bail!(
                "Hive crashed during startup (exit code: {})\n\nStderr:\n{}",
                status,
                stderr_content
            );
        }
        Ok(None) => {
            // Process still running - good!
            n!("start_hive_alive", "✅ Process still alive (PID: {})", pid);
        }
        Err(e) => {
            anyhow::bail!("Failed to check process status: {}", e);
        }
    }

    // TEAM-291: Verify HTTP server is responding
    n!("start_hive_health_check", "🏥 Checking health endpoint: http://localhost:{}/health", port);

    let health_url = format!("http://localhost:{}/health", port);
    let client = reqwest::Client::new();
    
    match client.get(&health_url).timeout(std::time::Duration::from_secs(3)).send().await {
        Ok(response) if response.status().is_success() => {
            n!("start_hive_health_ok", "✅ Health check passed");
        }
        Ok(response) => {
            let stderr_content = std::fs::read_to_string(&stderr_path)
                .unwrap_or_else(|_| "(failed to read stderr)".to_string());
            
            n!("hive_health_error", "❌ Health check failed with status: {}", response.status());
            n!("hive_crash_stderr", "Stderr output:\n{}", stderr_content);
            
            anyhow::bail!(
                "Hive HTTP server returned error status: {}\n\nStderr:\n{}",
                response.status(),
                stderr_content
            );
        }
        Err(e) => {
            let stderr_content = std::fs::read_to_string(&stderr_path)
                .unwrap_or_else(|_| "(failed to read stderr)".to_string());
            
            n!("hive_health_timeout", "❌ Health check failed: {}", e);
            n!("hive_crash_stderr", "Stderr output:\n{}", stderr_content);
            
            anyhow::bail!(
                "Hive HTTP server not responding: {}\n\nStderr:\n{}",
                e,
                stderr_content
            );
        }
    }

    // TEAM-291: Clean up stderr file on success
    let _ = std::fs::remove_file(&stderr_path);

    n!("start_hive_complete", "✅ Hive started at 'http://localhost:{}'", port);

    Ok(())
}

/// Start rbee-hive remotely via SSH
///
/// TEAM-292: Added queen_url parameter
/// TEAM-314: Made queen_url optional for remote hives + added detailed error handling
async fn start_hive_remote(host: &str, install_dir: &str, port: u16, queen_url: Option<&str>) -> Result<()> {
    use observability_narration_core::n;
    
    let queen_display = queen_url.unwrap_or("none");
    n!("start_hive_remote", "▶️  Starting rbee-hive on '{}' via SSH (queen: {})...", host, queen_display);

    let client = SshClient::connect(host).await?;
    let remote_path = format!("{}/rbee-hive", install_dir);

    // TEAM-314: Check if binary exists before trying to start
    n!("hive_binary_check", "🔍 Checking if hive binary exists at {}", remote_path);
    let binary_check = client
        .execute(&format!("test -f {} && echo 'exists' || echo 'missing'", remote_path))
        .await;
    
    match binary_check {
        Ok(output) if output.trim() == "missing" => {
            n!("hive_binary_missing", "❌ Hive binary not found at {}", remote_path);
            anyhow::bail!(
                "Hive binary not found at {} on '{}'. Run 'rbee hive install -a {}' first.",
                remote_path, host, host
            );
        }
        Ok(_) => {
            n!("hive_binary_found", "✅ Hive binary found at {}", remote_path);
        }
        Err(e) => {
            n!("hive_binary_check_failed", "⚠️  Failed to check binary: {}", e);
            // Continue anyway - might still work
        }
    }

    // Create stderr capture file on remote
    let stderr_path = format!("/tmp/rbee-hive-{}.stderr", std::process::id());
    n!("hive_stderr_setup", "📄 Setting up stderr capture at {}", stderr_path);

    // Start hive in background with stderr capture (not /dev/null)
    // TEAM-314: Capture stderr for diagnostics, similar to queen-lifecycle
    let command = if let Some(url) = queen_url {
        format!(
            "nohup {} --port {} --queen-url {} --hive-id {} > /dev/null 2>{} &",
            remote_path, port, url, host, stderr_path
        )
    } else {
        // No queen URL - hive runs standalone
        format!(
            "nohup {} --port {} --hive-id {} > /dev/null 2>{} &",
            remote_path, port, host, stderr_path
        )
    };
    
    n!("hive_spawn", "🚀 Spawning hive process on '{}'", host);
    client
        .execute(&command)
        .await
        .context("Failed to execute hive start command")?;

    // Wait for startup
    n!("hive_startup_wait", "⏳ Waiting for hive to start (2 seconds)...");
    tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;

    // Verify it's running
    n!("hive_process_check", "🔍 Checking if hive process is running...");
    let is_running = client
        .execute("pgrep -f rbee-hive")
        .await
        .is_ok();

    if !is_running {
        // TEAM-314: Fetch stderr for diagnostics
        n!("hive_crash_detected", "❌ Hive process not running - fetching error logs...");
        
        let stderr_content = client
            .execute(&format!("cat {} 2>/dev/null || echo 'No stderr file found'", stderr_path))
            .await
            .unwrap_or_else(|_| "Failed to read stderr".to_string());
        
        // Clean up stderr file
        let _ = client.execute(&format!("rm -f {}", stderr_path)).await;
        
        n!("hive_crash_stderr", "Stderr output:\n{}", stderr_content);
        
        anyhow::bail!(
            "Hive failed to start on '{}'\n\nStderr:\n{}",
            host,
            stderr_content
        );
    }
    
    n!("hive_process_alive", "✅ Hive process is running on '{}'", host);
    
    // Clean up stderr file on success
    let _ = client.execute(&format!("rm -f {}", stderr_path)).await;

    n!("start_hive_complete", "✅ Hive started on '{}'", host);

    Ok(())
}
