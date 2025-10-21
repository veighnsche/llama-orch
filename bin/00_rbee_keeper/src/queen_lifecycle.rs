//! Queen-rbee lifecycle management
//!
//! Manages the lifecycle of queen-rbee daemon from rbee-keeper CLI.
//!
//! This module uses the shared `daemon-lifecycle` crate for common lifecycle operations.
//! All observability is handled through narration-core (no tracing).
//!
//! TEAM-185: Consolidated from separate rbee-keeper-queen-lifecycle crate into this module
//! TEAM-185: Imports ACTION_QUEEN_START and ACTION_QUEEN_STOP from operations module

use anyhow::{Context, Result};
use daemon_lifecycle::DaemonManager;
use observability_narration_core::NarrationFactory;
use rbee_config::RbeeConfig; // TEAM-195: For preflight validation
use std::time::Duration;
use timeout_enforcer::TimeoutEnforcer;
use tokio::time::sleep;

// TEAM-192: Local narration factory for queen lifecycle
const NARRATE: NarrationFactory = NarrationFactory::new("kpr-life");

/// Handle to the queen-rbee process
///
/// Tracks whether rbee-keeper started the queen and provides cleanup.
/// IMPORTANT: Only shuts down queen if rbee-keeper started it!
pub struct QueenHandle {
    /// True if rbee-keeper started the queen (must cleanup)
    /// False if queen was already running (don't touch it)
    started_by_us: bool,

    /// Base URL of the queen
    base_url: String,

    /// Process ID if we started it
    pid: Option<u32>,
}

impl QueenHandle {
    /// Create handle for queen that was already running
    fn already_running(base_url: String) -> Self {
        Self { started_by_us: false, base_url, pid: None }
    }

    /// Create handle for queen that we just started
    fn started_by_us(base_url: String, pid: Option<u32>) -> Self {
        Self { started_by_us: true, base_url, pid }
    }

    /// Check if we started the queen (and should clean it up)
    pub fn should_cleanup(&self) -> bool {
        self.started_by_us
    }

    /// Get the queen's base URL
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Keep the queen alive (no shutdown after task)
    ///
    /// Queen stays running for future tasks. After 5 minutes of inactivity,
    /// the hive will automatically purge workers (handled by rbee-hive).
    ///
    /// # Returns
    /// * `Ok(())` - Always succeeds (queen stays alive)
    pub async fn shutdown(self) -> Result<()> {
        NARRATE
            .action("queen_stop")
            .human("Task complete, keeping queen alive for future tasks")
            .emit();
        Ok(())
    }
}

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
/// * `Err` - Failed to start queen or timeout waiting for health
///
/// TEAM-163: Added 30-second total timeout with visual countdown
pub async fn ensure_queen_running(base_url: &str) -> Result<QueenHandle> {
    // TEAM-163: Use TimeoutEnforcer for hard timeout with visual countdown
    TimeoutEnforcer::new(Duration::from_secs(30))
        .with_label("Starting queen-rbee")
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

/// Check if queen is healthy by calling /health endpoint
///
/// # Arguments
/// * `base_url` - Queen URL (e.g., "http://localhost:8500")
///
/// # Returns
/// * `Ok(true)` - Queen is running and healthy
/// * `Ok(false)` - Queen is not running (connection refused)
/// * `Err` - Other errors (timeout, invalid response, etc.)
async fn is_queen_healthy(base_url: &str) -> Result<bool> {
    let health_url = format!("{}/health", base_url);

    let client = reqwest::Client::builder().timeout(Duration::from_millis(500)).build()?;

    match client.get(&health_url).send().await {
        Ok(response) => {
            if response.status().is_success() {
                Ok(true)
            } else {
                Ok(false)
            }
        }
        Err(e) => {
            // Connection refused means queen is not running
            if e.is_connect() {
                Ok(false)
            } else {
                Err(e.into())
            }
        }
    }
}

/// Poll health endpoint until queen is ready
///
/// Uses exponential backoff: 100ms, 200ms, 400ms, 800ms, 1600ms, 3200ms
///
/// # Arguments
/// * `base_url` - Queen URL
/// * `timeout` - Maximum time to wait
///
/// # Returns
/// * `Ok(())` - Queen became healthy
/// * `Err` - Timeout or other error
async fn poll_until_healthy(base_url: &str, timeout: Duration) -> Result<()> {
    let start = std::time::Instant::now();
    let mut delay = Duration::from_millis(100);
    let max_delay = Duration::from_millis(3200);
    let mut attempt = 0u32;

    loop {
        attempt += 1;

        // Check if we've exceeded timeout
        if start.elapsed() >= timeout {
            NARRATE
                .action("queen_start")
                .context(timeout.as_secs().to_string())
                .human("Queen failed to become healthy within {} seconds")
                .error_kind("startup_timeout")
                .emit();
            anyhow::bail!("Timeout waiting for queen to become healthy");
        }

        // Try health check
        match is_queen_healthy(base_url).await {
            Ok(true) => {
                NARRATE
                    .action("queen_poll")
                    .context(format!("{:?}", start.elapsed()))
                    .human("Queen health check succeeded after {}")
                    .emit();
                return Ok(());
            }
            Ok(false) => {
                NARRATE
                    .action("queen_poll")
                    .context(attempt.to_string())
                    .context(delay.as_millis().to_string())
                    .human("Polling queen health (attempt {}, delay {}ms)")
                    .emit();
            }
            Err(e) => {
                NARRATE
                    .action("queen_poll")
                    .context(e.to_string())
                    .human("Queen health check failed: {}")
                    .error_kind("health_check_failed")
                    .emit();
            }
        }

        // Wait before next attempt
        sleep(delay).await;

        // Exponential backoff (cap at max_delay)
        delay = std::cmp::min(delay * 2, max_delay);
    }
}
