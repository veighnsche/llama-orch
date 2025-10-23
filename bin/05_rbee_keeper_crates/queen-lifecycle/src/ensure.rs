//! Ensure queen-rbee is running
//!
//! TEAM-259: Extracted from rbee-keeper/src/queen_lifecycle.rs

use anyhow::{Context, Result};
use daemon_lifecycle::DaemonManager;
use observability_narration_core::NarrationFactory;
use rbee_config::RbeeConfig;
use std::time::Duration;
use timeout_enforcer::TimeoutEnforcer;

use crate::health::{is_queen_healthy, poll_until_healthy};
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
    let start_time = std::time::Instant::now();

    // Step 1: Check if queen is already running
    if is_queen_healthy(base_url).await? {
        NARRATE.action("queen_check").human("Queen is already running and healthy").emit();
        return Ok(QueenHandle::already_running(base_url.to_string()));
    }

    // Step 2: TEAM-195: Preflight validation before starting queen
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

    // Step 3: Queen is not running, start it
    NARRATE.action("queen_start").human("⚠️  Queen is asleep, waking queen").emit();

    // Step 4: Find queen-rbee binary in target directory
    let queen_binary = DaemonManager::find_in_target("queen-rbee")
        .context("Failed to find queen-rbee binary in target directory")?;

    NARRATE
        .action("queen_start")
        .context(queen_binary.display().to_string())
        .human("Found queen-rbee binary at {}")
        .emit();

    // Step 5: Spawn queen process
    let args = vec!["--port".to_string(), "8500".to_string()];
    let manager = DaemonManager::new(queen_binary, args);

    let mut _child = manager.spawn().await.context("Failed to spawn queen-rbee process")?;

    NARRATE
        .action("queen_start")
        .human("Queen-rbee process spawned, waiting for health check")
        .emit();

    // Step 6: Poll health until ready (30 second timeout)
    poll_until_healthy(base_url, Duration::from_secs(30))
        .await
        .context("Queen failed to become healthy within timeout")?;

    // Step 7: Success!
    let elapsed_ms = start_time.elapsed().as_millis() as u64;
    let pid = _child.id();
    NARRATE
        .action("queen_ready")
        .human("✅ Queen is awake and healthy")
        .duration_ms(elapsed_ms)
        .emit();

    Ok(QueenHandle::started_by_us(base_url.to_string(), pid))
}
