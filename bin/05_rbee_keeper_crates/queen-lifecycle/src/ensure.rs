//! Ensure queen-rbee is running
//!
//! TEAM-259: Extracted from rbee-keeper/src/queen_lifecycle.rs
//! TEAM-276: Refactored to use daemon-lifecycle::ensure_daemon_with_handle

use anyhow::{Context, Result};
use daemon_lifecycle::{
    ensure_daemon_with_handle, poll_until_healthy, DaemonManager, HealthPollConfig,
};
use observability_narration_core::NarrationFactory;
use rbee_config::RbeeConfig;
use std::time::Duration;
use timeout_enforcer::TimeoutEnforcer;

use crate::types::QueenHandle;

// TEAM-192: Local narration factory for queen lifecycle
const NARRATE: NarrationFactory = NarrationFactory::new("kpr-life");

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

    ensure_daemon_with_handle(
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
    .await
}

/// Spawn queen with preflight checks
///
/// TEAM-276: Extracted spawn logic to use with ensure_daemon_with_handle
async fn spawn_queen_with_preflight(base_url: &str) -> Result<()> {
    // Step 1: TEAM-195: Preflight validation before starting queen
    NARRATE.action("queen_preflight").human("📋 Loading rbee configuration...").emit();

    let config = RbeeConfig::load().context("Failed to load rbee config")?;

    NARRATE
        .action("queen_preflight")
        .human(format!("✅ Config loaded from {}", RbeeConfig::config_dir()?.display()))
        .emit();

    // Validate configuration
    NARRATE.action("queen_preflight").human("🔍 Validating configuration...").emit();

    let validation_result = config.validate().context("Configuration validation failed")?;

    if !validation_result.is_valid() {
        NARRATE
            .action("queen_preflight")
            .human(format!(
                "❌ Configuration validation failed:\n\n{}\n\nPlease fix the errors in ~/.config/rbee/ and try again.",
                validation_result.errors.join("\n")
            ))
            .error_kind("config_validation_failed")
            .emit();
        anyhow::bail!("Configuration validation failed: {}", validation_result.errors.join(", "));
    }

    // Report hive count
    let hive_count = config.hives.len();
    NARRATE.action("queen_preflight").human(format!("✅ {} hive(s) configured", hive_count)).emit();

    // Report capabilities
    let caps_count = config.capabilities.aliases().len();
    if caps_count > 0 {
        NARRATE
            .action("queen_preflight")
            .human(format!("📊 {} hive(s) have cached capabilities", caps_count))
            .emit();
    } else {
        NARRATE
            .action("queen_preflight")
            .human("⚠️  No cached capabilities found (hives not yet started)")
            .emit();
    }

    // Report warnings if any
    if validation_result.has_warnings() {
        for warning in &validation_result.warnings {
            NARRATE.action("queen_preflight").human(format!("⚠️  {}", warning)).emit();
        }
    }

    NARRATE.action("queen_preflight").human("✅ All preflight checks passed").emit();

    // Step 2: Find queen-rbee binary in target directory
    let queen_binary = DaemonManager::find_in_target("queen-rbee")
        .context("Failed to find queen-rbee binary in target directory")?;

    NARRATE
        .action("queen_start")
        .context(queen_binary.display().to_string())
        .human("Found queen-rbee binary at {}")
        .emit();

    // Step 3: Spawn queen process
    let args = vec!["--port".to_string(), "8500".to_string()];
    let manager = DaemonManager::new(queen_binary, args);

    let child = manager.spawn().await.context("Failed to spawn queen-rbee process")?;

    NARRATE.action("queen_spawned").human("Queen-rbee process spawned, polling health...").emit();

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
