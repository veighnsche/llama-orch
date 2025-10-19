//! Queen-rbee lifecycle management
//!
//! TEAM-085: Shared utility for ensuring queen-rbee is running
//!
//! ⚠️ CRITICAL: Only use this for commands that REQUIRE queen-rbee to exist!
//!
//! Commands that NEED queen-rbee (orchestration):
//! - infer (routes to remote nodes)
//! - setup add-node (registers nodes in queen-rbee)
//! - setup list-nodes (queries queen-rbee registry)
//! - setup remove-node (removes from queen-rbee registry)
//!
//! Commands that DON'T need queen-rbee (direct operations):
//! - logs (reads logs from remote node directly via SSH)
//! - workers list (queries remote rbee-hive directly)
//! - workers health (queries remote rbee-hive directly)
//! - workers shutdown (sends shutdown to remote rbee-hive directly)
//! - hive commands (SSH to rbee-hive, no orchestration)
//! - install (local file operations)

use anyhow::Result;
use colored::Colorize;
use std::time::Duration;

/// Ensure queen-rbee is running, auto-start if needed
///
/// This is the SINGLE SOURCE OF TRUTH for queen-rbee lifecycle management.
/// All rbee-keeper commands that talk to queen-rbee MUST call this first.
///
/// # Arguments
/// * `client` - HTTP client to use for health checks
/// * `queen_url` - Queen-rbee URL (typically "http://localhost:8080")
///
/// # Returns
/// * `Ok(())` if queen-rbee is running or was successfully started
/// * `Err` if queen-rbee failed to start
pub async fn ensure_queen_rbee_running(client: &reqwest::Client, queen_url: &str) -> Result<()> {
    // Check if queen-rbee is already running
    let health_url = format!("{}/health", queen_url);

    match client.get(&health_url).timeout(Duration::from_millis(500)).send().await {
        Ok(resp) if resp.status().is_success() => {
            // Already running, no need to print anything
            return Ok(());
        }
        _ => {
            println!("{}", "⚠️  queen-rbee not running, starting...".yellow());
        }
    }

    // Find queen-rbee binary
    let queen_binary = std::env::current_exe()?
        .parent()
        .ok_or_else(|| anyhow::anyhow!("Cannot find binary directory"))?
        .join("queen-rbee");

    if !queen_binary.exists() {
        anyhow::bail!(
            "queen-rbee binary not found at {:?}. Run: cargo build --bin queen-rbee",
            queen_binary
        );
    }

    // Start queen-rbee as background process
    println!("{}", "🚀 Starting queen-rbee daemon...".cyan());

    // TEAM-085: Use temp database for ephemeral mode
    let temp_db = std::env::temp_dir().join("queen-rbee-ephemeral.db");

    // TEAM-088: CRITICAL FIX - Don't silence logs! We need to see what's happening!
    // Use RBEE_SILENT=1 to suppress logs if needed
    let (stdout_cfg, stderr_cfg) = if std::env::var("RBEE_SILENT").is_ok() {
        (std::process::Stdio::null(), std::process::Stdio::null())
    } else {
        (std::process::Stdio::inherit(), std::process::Stdio::inherit())
    };

    let mut child = tokio::process::Command::new(&queen_binary)
        .arg("--port")
        .arg("8080")
        .arg("--database")
        .arg(&temp_db)
        .stdout(stdout_cfg)
        .stderr(stderr_cfg)
        .spawn()?;

    // Wait for queen-rbee to be ready (max 30 seconds)
    for attempt in 0..300 {
        tokio::time::sleep(Duration::from_millis(100)).await;

        // Check if process died first
        if let Ok(Some(status)) = child.try_wait() {
            anyhow::bail!("queen-rbee exited with status: {}", status);
        }

        match client.get(&health_url).timeout(Duration::from_millis(500)).send().await {
            Ok(resp) if resp.status().is_success() => {
                println!("{}", "✓ queen-rbee started successfully".green());

                // Detach the child process so it keeps running
                let _ = child.id();
                std::mem::forget(child);

                return Ok(());
            }
            Ok(resp) => {
                if attempt % 10 == 0 && attempt > 0 {
                    println!(
                        "{}",
                        format!(
                            "  ⏳ queen-rbee returned HTTP {}, waiting... ({}/30s)",
                            resp.status(),
                            attempt / 10
                        )
                        .dimmed()
                    );
                }
            }
            Err(_) if attempt % 10 == 0 && attempt > 0 => {
                println!(
                    "{}",
                    format!("  ⏳ Waiting for queen-rbee to start... ({}/30s)", attempt / 10)
                        .dimmed()
                );
            }
            _ => {}
        }
    }

    anyhow::bail!("queen-rbee failed to start within 30 seconds")
}
