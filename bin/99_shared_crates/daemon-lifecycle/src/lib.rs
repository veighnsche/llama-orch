// TEAM-135: Created by TEAM-135 (scaffolding)
// TEAM-152: Implemented core daemon spawning functionality
// TEAM-152: Replaced tracing with narration for observability
// Purpose: Shared daemon lifecycle management for rbee-keeper, queen-rbee, and rbee-hive

#![warn(missing_docs)]
#![warn(clippy::all)]

//! daemon-lifecycle
//!
//! **Category:** Utility
//! **Pattern:** Function-based
//! **Standard:** See `/bin/CRATE_INTERFACE_STANDARD.md`
//!
//! Shared daemon lifecycle management functionality for managing daemon processes
//! across rbee-keeper, queen-rbee, and rbee-hive binaries.
//! All observability is handled through narration-core (no tracing).
//!
//! # Interface
//!
//! ## Utility Functions
//! ```rust,no_run
//! use daemon_lifecycle::{DaemonManager, spawn_daemon};
//! use std::path::PathBuf;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Create daemon manager
//! let manager = DaemonManager::new(
//!     PathBuf::from("target/debug/queen-rbee"),
//!     vec!["--config".to_string(), "config.toml".to_string()]
//! );
//!
//! // Spawn daemon process
//! let child = manager.spawn().await?;
//!
//! // Find binary in target directory
//! let binary_path = DaemonManager::find_in_target("queen-rbee")?;
//! # Ok(())
//! # }
//! ```

use anyhow::{Context, Result};
use observability_narration_core::NarrationFactory;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::process::{Child, Command};

// TEAM-197: Migrated to narration-core v0.5.0 pattern
// - Changed from Narration::new() to NarrationFactory pattern
// - Shortened actor from "⚙️ daemon-lifecycle" to "dmn-life" (8 chars, ≤10 limit)
// - Using .action() instead of .narrate()
// - Fixed-width format for better log readability
const NARRATE: NarrationFactory = NarrationFactory::new("dmn-life");

/// Daemon manager for spawning and managing daemon processes
pub struct DaemonManager {
    binary_path: PathBuf,
    args: Vec<String>,
}

impl DaemonManager {
    /// Create a new daemon manager
    ///
    /// # Arguments
    /// * `binary_path` - Path to the daemon binary
    /// * `args` - Command-line arguments for the daemon
    pub fn new(binary_path: PathBuf, args: Vec<String>) -> Self {
        Self { binary_path, args }
    }

    /// Spawn the daemon process
    ///
    /// Returns the spawned child process handle
    ///
    /// # Errors
    /// Returns error if:
    /// - Binary not found
    /// - Failed to spawn process
    pub async fn spawn(&self) -> Result<Child> {
        // TEAM-197: Updated to narration-core v0.5.0 pattern
        NARRATE
            .action("spawn")
            .context(self.binary_path.display().to_string())
            .context(format!("{:?}", self.args))
            .human("Spawning daemon: {0} with args: {1}")
            .emit();

        // ============================================================
        // BUG FIX: TEAM-164 | Daemon holds parent's pipes open
        // ============================================================
        // SUSPICION:
        // - E2E test hangs when using Command::output() to run rbee-keeper
        // - Direct command execution works fine
        //
        // INVESTIGATION:
        // - Tested with timeout - confirmed hang at Command::output()
        // - Tested with file redirection - works (exits immediately)
        // - Tested with pipes - hangs (never completes)
        // - Added TTY detection to timeout-enforcer - didn't fix it
        // - Checked daemon-lifecycle spawn code - FOUND IT!
        //
        // ROOT CAUSE:
        // - Daemon spawned with Stdio::inherit() inherits parent's file descriptors
        // - When parent runs via Command::output(), stdout/stderr are PIPES
        // - Daemon holds pipes open even after parent exits
        // - Command::output() waits for ALL pipe readers to close
        // - Result: infinite hang
        //
        // FIX:
        // - Use Stdio::null() for daemon stdout/stderr
        // - Daemon no longer holds parent's pipes
        // - Parent can exit immediately
        // - Command::output() completes as expected
        //
        // TESTING:
        // - cargo xtask e2e:queen (was hanging, now passes)
        // - Direct: target/debug/rbee-keeper queen start (still works)
        // - With output capture: works (no more hang)
        // ============================================================

        // TEAM-189: Propagate SSH agent environment variables to daemon
        // This allows the daemon to use the parent's SSH agent for authentication
        let mut cmd = Command::new(&self.binary_path);
        cmd.args(&self.args)
            .stdout(Stdio::null()) // TEAM-164: Don't inherit parent's stdout pipe
            .stderr(Stdio::null()); // TEAM-164: Don't inherit parent's stderr pipe

        // Propagate SSH agent socket if available
        if let Ok(ssh_auth_sock) = std::env::var("SSH_AUTH_SOCK") {
            cmd.env("SSH_AUTH_SOCK", ssh_auth_sock);
        }

        let child = cmd
            .spawn()
            .context(format!("Failed to spawn daemon: {}", self.binary_path.display()))?;

        let pid_str = child.id().map(|p| p.to_string()).unwrap_or_else(|| "unknown".to_string());
        // TEAM-197: Updated to narration-core v0.5.0 pattern
        NARRATE
            .action("spawned")
            .context(pid_str.clone())
            .human("Daemon spawned with PID: {}")
            .emit();
        Ok(child)
    }

    /// Find a binary in the target directory (for development)
    /// Tries to find a binary in the standard Cargo target directory:
    /// - `target/debug/{name}`
    /// - `target/release/{name}`
    ///
    /// # Arguments
    /// * `name` - Binary name (e.g., "queen-rbee")
    ///
    /// # Returns
    /// Path to the binary if found
    pub fn find_in_target(name: &str) -> Result<PathBuf> {
        // TEAM-255: Find workspace root by looking for Cargo.toml
        let mut current = std::env::current_dir()?;
        let workspace_root = loop {
            if current.join("Cargo.toml").exists() && 
               (current.join("xtask").exists() || current.join("bin").exists()) {
                break current;
            }
            if let Some(parent) = current.parent() {
                current = parent.to_path_buf();
            } else {
                // Fallback to current dir if workspace root not found
                break std::env::current_dir()?;
            }
        };

        // Try debug first (development mode)
        let debug_path = workspace_root.join("target/debug").join(name);
        if debug_path.exists() {
            // TEAM-197: Updated to narration-core v0.5.0 pattern
            NARRATE
                .action("find_binary")
                .context(name.to_string())
                .context(debug_path.display().to_string())
                .human("Found binary '{0}' at: {1}")
                .emit();
            return Ok(debug_path);
        }

        // Try release
        let release_path = workspace_root.join("target/release").join(name);
        if release_path.exists() {
            // TEAM-197: Updated to narration-core v0.5.0 pattern
            NARRATE
                .action("find_binary")
                .context(name.to_string())
                .context(release_path.display().to_string())
                .human("Found binary '{0}' at: {1}")
                .emit();
            return Ok(release_path);
        }

        // TEAM-197: Updated to narration-core v0.5.0 pattern
        NARRATE
            .action("find_binary")
            .context(name.to_string())
            .human("Binary '{}' not found in target/debug or target/release")
            .error_kind("binary_not_found")
            .emit_error();
        anyhow::bail!("Binary '{}' not found in target/debug or target/release", name)
    }
}

/// Helper function to spawn a daemon with default settings
///
/// # Arguments
/// * `binary_path` - Path to the daemon binary
/// * `args` - Command-line arguments
///
/// # Returns
/// Spawned child process
pub async fn spawn_daemon<P: AsRef<Path>>(binary_path: P, args: Vec<String>) -> Result<Child> {
    let manager = DaemonManager::new(binary_path.as_ref().to_path_buf(), args);
    manager.spawn().await
}

// ============================================================================
// TEAM-259: "Ensure Daemon Running" Pattern
// ============================================================================
// Extracted from rbee-keeper::ensure_queen_running and queen-rbee::ensure_hive_running
// This pattern is used at multiple levels:
// - rbee-keeper → queen-rbee
// - queen-rbee → rbee-hive
// - (future) rbee-hive → llm-worker

use std::time::Duration;

/// Check if daemon is healthy via HTTP health endpoint
///
/// TEAM-259: Extracted from is_queen_healthy and is_hive_healthy
///
/// # Arguments
/// * `base_url` - Base URL of daemon (e.g., "http://localhost:8500")
/// * `health_endpoint` - Health endpoint path (default: "/health")
/// * `timeout` - HTTP timeout (default: 2 seconds)
///
/// # Returns
/// * `true` - Daemon is healthy
/// * `false` - Daemon is not responding or unhealthy
pub async fn is_daemon_healthy(
    base_url: &str,
    health_endpoint: Option<&str>,
    timeout: Option<Duration>,
) -> bool {
    let endpoint = health_endpoint.unwrap_or("/health");
    let timeout = timeout.unwrap_or(Duration::from_secs(2));

    let client = match reqwest::Client::builder().timeout(timeout).build() {
        Ok(c) => c,
        Err(_) => return false,
    };

    let url = format!("{}{}", base_url, endpoint);

    match client.get(&url).send().await {
        Ok(response) if response.status().is_success() => true,
        _ => false,
    }
}

/// Ensure daemon is running, auto-start if needed
///
/// TEAM-259: Extracted from ensure_queen_running and ensure_hive_running
///
/// This function implements the "ensure daemon running" pattern:
/// 1. Check if daemon is healthy (HTTP /health endpoint)
/// 2. If not running, spawn daemon using provided callback
/// 3. Wait for health check to pass (with timeout)
///
/// # Arguments
/// * `daemon_name` - Name of daemon (for logging)
/// * `base_url` - Base URL of daemon (e.g., "http://localhost:8500")
/// * `job_id` - Optional job ID for narration routing
/// * `spawn_fn` - Async function to spawn the daemon
/// * `timeout` - Max time to wait for health (default: 30 seconds)
/// * `poll_interval` - Health check interval (default: 500ms)
///
/// # Returns
/// * `Ok(true)` - Daemon was already running
/// * `Ok(false)` - Daemon was started by us
/// * `Err` - Failed to start daemon or timeout
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::ensure_daemon_running;
/// use anyhow::Result;
///
/// # async fn example() -> Result<()> {
/// ensure_daemon_running(
///     "queen-rbee",
///     "http://localhost:8500",
///     None,
///     || async {
///         // Spawn queen daemon here
///         Ok(())
///     },
///     None,
///     None,
/// ).await?;
/// # Ok(())
/// # }
/// ```
pub async fn ensure_daemon_running<F, Fut>(
    daemon_name: &str,
    base_url: &str,
    job_id: Option<&str>,
    spawn_fn: F,
    timeout: Option<Duration>,
    poll_interval: Option<Duration>,
) -> Result<bool>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = Result<()>>,
{
    let timeout = timeout.unwrap_or(Duration::from_secs(30));
    let poll_interval = poll_interval.unwrap_or(Duration::from_millis(500));

    // Check if daemon is already healthy
    if is_daemon_healthy(base_url, None, None).await {
        let mut narration = NARRATE.action("daemon_check").context(daemon_name);
        if let Some(jid) = job_id {
            narration = narration.job_id(jid);
        }
        narration.human(&format!("{} is already running", daemon_name)).emit();
        return Ok(true); // Already running
    }

    // Daemon is not running, start it
    let mut narration = NARRATE.action("daemon_start").context(daemon_name);
    if let Some(jid) = job_id {
        narration = narration.job_id(jid);
    }
    narration.human(&format!("⚠️  {} is not running, starting...", daemon_name)).emit();

    // Spawn daemon
    spawn_fn().await?;

    // Wait for daemon to become healthy (with timeout)
    let start_time = std::time::Instant::now();

    loop {
        if is_daemon_healthy(base_url, None, None).await {
            let mut narration = NARRATE.action("daemon_start").context(daemon_name);
            if let Some(jid) = job_id {
                narration = narration.job_id(jid);
            }
            narration.human(&format!("✅ {} is now running and healthy", daemon_name)).emit();
            return Ok(false); // Started by us
        }

        if start_time.elapsed() > timeout {
            return Err(anyhow::anyhow!(
                "Timeout waiting for {} to become healthy (waited {:?})",
                daemon_name,
                timeout
            ));
        }

        tokio::time::sleep(poll_interval).await;
    }
}
