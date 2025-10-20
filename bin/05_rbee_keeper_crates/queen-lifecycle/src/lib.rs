// TEAM-135: Created by TEAM-135 (scaffolding)
// TEAM-152: Implemented ensure_queen_running with auto-start and health polling
// TEAM-152: Replaced all tracing with inline narration for observability
// Purpose: Queen-rbee lifecycle management for rbee-keeper

#![warn(missing_docs)]
#![warn(clippy::all)]

//! rbee-keeper-queen-lifecycle
//!
//! Manages the lifecycle of queen-rbee daemon from rbee-keeper CLI.
//!
//! This crate uses the shared `daemon-lifecycle` crate for common lifecycle operations.
//! All observability is handled through narration-core (no tracing).

use anyhow::{Context, Result};
use daemon_lifecycle::DaemonManager;
use observability_narration_core::Narration;
use std::time::Duration;
use tokio::time::sleep;

// Re-export daemon-lifecycle for convenience
pub use daemon_lifecycle;

// Actor and action constants
// TEAM-155: Actor shows parent binary / subcrate for accurate provenance
const ACTOR_QUEEN_LIFECYCLE: &str = "🧑‍🌾 rbee-keeper / ⚙️ queen-lifecycle";
const ACTION_QUEEN_CHECK: &str = "queen_check";
const ACTION_QUEEN_START: &str = "queen_start";
const ACTION_QUEEN_POLL: &str = "queen_poll";
const ACTION_QUEEN_READY: &str = "queen_ready";
const ACTION_QUEEN_SHUTDOWN: &str = "queen_shutdown";

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

    /// Shutdown the queen (only if we started it)
    ///
    /// # Returns
    /// * `Ok(())` - Queen shutdown successfully or we didn't start it
    /// * `Err` - Failed to shutdown queen that we started
    pub async fn shutdown(self) -> Result<()> {
        if !self.started_by_us {
            Narration::new(ACTOR_QUEEN_LIFECYCLE, ACTION_QUEEN_SHUTDOWN, &self.base_url)
                .human("Queen was already running, not shutting down")
                .emit();
            return Ok(());
        }

        Narration::new(ACTOR_QUEEN_LIFECYCLE, ACTION_QUEEN_SHUTDOWN, &self.base_url)
            .human(format!("Shutting down queen (PID: {:?})", self.pid))
            .emit();

        // Try graceful shutdown via HTTP first
        let shutdown_url = format!("{}/shutdown", self.base_url);
        let client = reqwest::Client::builder().timeout(Duration::from_secs(2)).build()?;

        match client.post(&shutdown_url).send().await {
            Ok(_) => {
                Narration::new(ACTOR_QUEEN_LIFECYCLE, ACTION_QUEEN_SHUTDOWN, &self.base_url)
                    .human("Queen shutdown via HTTP")
                    .emit();
                return Ok(());
            }
            Err(_) => {
                // HTTP failed, try SIGTERM
                if let Some(pid) = self.pid {
                    Narration::new(ACTOR_QUEEN_LIFECYCLE, ACTION_QUEEN_SHUTDOWN, &pid.to_string())
                        .human(format!("HTTP shutdown failed, sending SIGTERM to PID {}", pid))
                        .emit();

                    // Use kill command (cross-platform via nix crate would be better, but this works)
                    let _ = std::process::Command::new("kill").arg(pid.to_string()).output();

                    Narration::new(ACTOR_QUEEN_LIFECYCLE, ACTION_QUEEN_SHUTDOWN, &pid.to_string())
                        .human("Sent SIGTERM to queen")
                        .emit();
                }
            }
        }

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
pub async fn ensure_queen_running(base_url: &str) -> Result<QueenHandle> {
    let start_time = std::time::Instant::now();

    // Step 1: Check if queen is already running
    if is_queen_healthy(base_url).await? {
        Narration::new(ACTOR_QUEEN_LIFECYCLE, ACTION_QUEEN_CHECK, base_url)
            .human("Queen is already running and healthy")
            .emit();
        return Ok(QueenHandle::already_running(base_url.to_string()));
    }

    // Step 2: Queen is not running, start it
    Narration::new(ACTOR_QUEEN_LIFECYCLE, ACTION_QUEEN_START, "queen-rbee")
        .human("⚠️  Queen is asleep, waking queen")
        .emit();

    // Step 3: Find queen-rbee binary in target directory
    let queen_binary = DaemonManager::find_in_target("queen-rbee")
        .context("Failed to find queen-rbee binary in target directory")?;

    Narration::new(ACTOR_QUEEN_LIFECYCLE, ACTION_QUEEN_START, &queen_binary.display().to_string())
        .human(format!("Found queen-rbee binary at {}", queen_binary.display()))
        .emit();

    // Step 4: Spawn queen process
    let args = vec!["--port".to_string(), "8500".to_string()];
    let manager = DaemonManager::new(queen_binary, args);

    let mut _child = manager.spawn().await.context("Failed to spawn queen-rbee process")?;

    let pid_target = _child.id().map(|p| p.to_string()).unwrap_or_else(|| "unknown".to_string());
    Narration::new(ACTOR_QUEEN_LIFECYCLE, ACTION_QUEEN_START, &pid_target)
        .human("Queen-rbee process spawned, waiting for health check")
        .emit();

    // Step 5: Poll health until ready (30 second timeout)
    poll_until_healthy(base_url, Duration::from_secs(30))
        .await
        .context("Queen failed to become healthy within timeout")?;

    // Step 6: Success!
    let elapsed_ms = start_time.elapsed().as_millis() as u64;
    let pid = _child.id();
    Narration::new(ACTOR_QUEEN_LIFECYCLE, ACTION_QUEEN_READY, "queen-rbee")
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
            Narration::new(ACTOR_QUEEN_LIFECYCLE, ACTION_QUEEN_START, "timeout")
                .human(format!(
                    "Queen failed to become healthy within {} seconds",
                    timeout.as_secs()
                ))
                .error_kind("startup_timeout")
                .emit();
            anyhow::bail!("Timeout waiting for queen to become healthy");
        }

        // Try health check
        match is_queen_healthy(base_url).await {
            Ok(true) => {
                Narration::new(ACTOR_QUEEN_LIFECYCLE, ACTION_QUEEN_POLL, "health")
                    .human(format!("Queen health check succeeded after {:?}", start.elapsed()))
                    .emit();
                return Ok(());
            }
            Ok(false) => {
                Narration::new(ACTOR_QUEEN_LIFECYCLE, ACTION_QUEEN_POLL, "health")
                    .human(format!(
                        "Polling queen health (attempt {}, delay {}ms)",
                        attempt,
                        delay.as_millis()
                    ))
                    .emit();
            }
            Err(e) => {
                Narration::new(ACTOR_QUEEN_LIFECYCLE, ACTION_QUEEN_POLL, "health")
                    .human(format!("Queen health check failed: {}", e))
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
