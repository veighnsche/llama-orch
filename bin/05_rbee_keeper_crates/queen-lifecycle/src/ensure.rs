//! Ensure queen-rbee is running
//!
//! TEAM-259: Extracted from rbee-keeper/src/queen_lifecycle.rs
//! TEAM-276: Refactored to use daemon-lifecycle::ensure_daemon_with_handle

use anyhow::{Context, Result};
use daemon_lifecycle::{
    ensure_daemon_with_handle, poll_until_healthy, DaemonManager, HealthPollConfig,
};
use observability_narration_core::n;
// TEAM-290: DELETED rbee_config import (file-based config deprecated)
use std::time::Duration;
use timeout_enforcer::TimeoutEnforcer;

use crate::types::QueenHandle;

/// Ensure queen-rbee is running, auto-start if needed
///
/// # Happy Flow (from a_human_wrote_this.md lines 11-19)
/// 1. Check health using HTTP GET /health
/// 2. If healthy → return Ok(())
/// 3. If not running:
///    - Print: "⚠️  queen is asleep, waking queen."
///    - Spawn queen-rbee process using daemon-lifecycle
///    - Poll health until ready (with timeout)
///    - Print: "✅ queen is awake and healthy."
///
/// # Arguments
/// * `base_url` - Queen URL (e.g., "http://localhost:8500")
///
/// # Returns
/// * `Ok(QueenHandle)` - Handle to queen (tracks if we started it for cleanup)
///
/// # Errors
///
/// Returns an error if queen fails to start or timeout waiting for health
///
/// TEAM-163: Added 30-second total timeout with visual countdown
pub async fn ensure_queen_running(base_url: &str) -> Result<QueenHandle> {
    // TEAM-197: Use TimeoutEnforcer with progress bar for visual feedback
    TimeoutEnforcer::new(Duration::from_secs(30))
        .with_label("Starting queen-rbee")
        .with_countdown() // Enable progress bar
        .enforce(ensure_queen_running_inner(base_url))
        .await
}

async fn ensure_queen_running_inner(base_url: &str) -> Result<QueenHandle> {
    // TEAM-276: Use shared ensure pattern from daemon-lifecycle
    let health_url = format!("{}/health", base_url);

    let handle = ensure_daemon_with_handle(
        "queen-rbee",
        &health_url,
        None,
        || async {
            // Spawn logic: preflight + start
            spawn_queen_with_preflight(base_url).await
        },
        || QueenHandle::already_running(base_url.to_string()),
        || QueenHandle::started_by_us(base_url.to_string(), None),
    )
    .await?;
    
    // TEAM-292: Fetch queen's actual URL from /v1/info endpoint
    // This allows queen to tell us its address (useful for future remote queens)
    match fetch_queen_url(base_url).await {
        Ok(queen_url) => {
            n!("queen_url_discovered", "✅ Queen URL discovered: {}", queen_url);
            Ok(handle.with_discovered_url(queen_url))
        }
        Err(e) => {
            // Fallback to base_url if discovery fails
            n!("queen_url_fallback", "⚠️  Failed to discover queen URL ({}), using default", e);
            Ok(handle)
        }
    }
}

/// Fetch queen's URL from /v1/info endpoint
///
/// TEAM-292: Service discovery - ask queen for its address
async fn fetch_queen_url(base_url: &str) -> Result<String> {
    let info_url = format!("{}/v1/info", base_url);
    let client = reqwest::Client::new();
    
    let response = client
        .get(&info_url)
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .context("Failed to fetch queen info")?;
    
    if !response.status().is_success() {
        anyhow::bail!("Queen info endpoint returned error: {}", response.status());
    }
    
    let info: serde_json::Value = response.json().await
        .context("Failed to parse queen info response")?;
    
    let queen_url = info["base_url"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Queen info missing base_url field"))?
        .to_string();
    
    Ok(queen_url)
}

/// Spawn queen with preflight checks
///
/// TEAM-276: Extracted spawn logic to use with ensure_daemon_with_handle
/// TEAM-290: Removed config loading (file-based config deprecated)
async fn spawn_queen_with_preflight(base_url: &str) -> Result<()> {
    // TEAM-290: No preflight validation (no config files)
    n!("queen_preflight", "✅ Localhost-only mode (no config needed)");

    // Step 2: Find queen-rbee binary
    // TEAM-296: ONLY use installed binary - no fallback to development binary
    // This ensures install/uninstall actually control whether queen can run
    let queen_binary = {
        let home = std::env::var("HOME").context("Failed to get HOME directory")?;
        let installed_path = std::path::PathBuf::from(format!("{}/.local/bin/queen-rbee", home));
        
        if installed_path.exists() {
            n!("queen_start", "Using installed queen-rbee binary at {}", installed_path.display());
            installed_path
        } else {
            n!("queen_start", "❌ Queen not installed. Run 'rbee queen install' first.");
            anyhow::bail!(
                "Queen not installed at {}. Run 'rbee queen install' to install from source.",
                installed_path.display()
            );
        }
    };

    // Step 3: Extract port from base_url and spawn queen process
    // base_url format: "http://localhost:7833"
    let port = base_url
        .split(':')
        .last()
        .context("Failed to extract port from base_url")?
        .to_string();
    
    let args = vec!["--port".to_string(), port];
    let manager = DaemonManager::new(queen_binary, args);

    let child = manager.spawn().await.context("Failed to spawn queen-rbee process")?;

    n!("queen_spawned", "Queen-rbee process spawned, polling health...");

    // Step 4: Poll health until ready
    poll_until_healthy(
        HealthPollConfig::new(base_url).with_daemon_name("queen-rbee").with_max_attempts(30),
    )
    .await
    .context("Queen failed to become healthy within timeout")?;

    // Keep child alive - detach from parent process
    // The child process will continue running independently
    drop(child);

    Ok(())
}
