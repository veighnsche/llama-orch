//! Daemon process manager
//!
//! TEAM-259: Extracted from lib.rs for better organization
//!
//! Provides DaemonManager for spawning and managing daemon processes.

use anyhow::{Context, Result};
use auto_update::AutoUpdater;
use observability_narration_core::n;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::process::{Child, Command};

/// Daemon manager for spawning and managing daemon processes
pub struct DaemonManager {
    binary_path: PathBuf,
    args: Vec<String>,
    auto_update: Option<(String, String)>, // (binary_name, source_dir)
}

impl DaemonManager {
    /// Create a new daemon manager
    ///
    /// # Arguments
    /// * `binary_path` - Path to the daemon binary
    /// * `args` - Command-line arguments for the daemon
    pub fn new(binary_path: PathBuf, args: Vec<String>) -> Self {
        Self { binary_path, args, auto_update: None }
    }

    /// Enable auto-update for this daemon
    ///
    /// When enabled, the daemon will be automatically rebuilt if any of its
    /// dependencies have changed before spawning.
    ///
    /// # Arguments
    /// * `binary_name` - Binary name (e.g., "queen-rbee")
    /// * `source_dir` - Source directory relative to workspace root (e.g., "bin/10_queen_rbee")
    ///
    /// # Example
    /// ```rust,no_run
    /// use daemon_lifecycle::DaemonManager;
    /// use std::path::PathBuf;
    ///
    /// # async fn example() -> anyhow::Result<()> {
    /// let manager = DaemonManager::new(
    ///     PathBuf::from("target/debug/queen-rbee"),
    ///     vec![]
    /// )
    /// .enable_auto_update("queen-rbee", "bin/10_queen_rbee");
    ///
    /// // Will auto-rebuild if dependencies changed
    /// let child = manager.spawn().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn enable_auto_update(
        mut self,
        binary_name: impl Into<String>,
        source_dir: impl Into<String>,
    ) -> Self {
        self.auto_update = Some((binary_name.into(), source_dir.into()));
        self
    }

    /// Spawn the daemon process
    ///
    /// If auto-update is enabled, checks if rebuild is needed and rebuilds before spawning.
    ///
    /// Returns the spawned child process handle
    ///
    /// # Errors
    /// Returns error if:
    /// - Binary not found
    /// - Failed to spawn process
    /// - Auto-update rebuild failed
    pub async fn spawn(&self) -> Result<Child> {
        // TEAM-259: Auto-update if enabled
        // TEAM-311: Migrated to n!() macro
        if let Some((binary_name, source_dir)) = &self.auto_update {
            n!("auto_update", "Checking if '{}' needs rebuild...", binary_name);

            let updater = AutoUpdater::new(binary_name, source_dir)?;

            if updater.needs_rebuild()? {
                n!("auto_rebuild", "🔨 Rebuilding '{}'...", binary_name);

                updater.rebuild()?;

                n!("auto_rebuild", "✅ Rebuild complete");
            } else {
                n!("auto_update", "✅ '{}' is up to date", binary_name);
            }
        }

        // TEAM-311: Migrated to n!() macro
        n!("spawn", "Spawning daemon: {} with args: {:?}", self.binary_path.display(), self.args);

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
        // TEAM-311: Migrated to n!() macro
        n!("spawned", "Daemon spawned with PID: {}", pid_str);
        Ok(child)
    }

    /// Find a binary (installed or development)
    ///
    /// TEAM-320: Checks installed location first, then falls back to development builds
    ///
    /// Search order:
    /// 1. `~/.local/bin/{name}` (installed)
    /// 2. `target/debug/{name}` (development)
    /// 3. `target/release/{name}` (development)
    ///
    /// # Arguments
    /// * `name` - Binary name (e.g., "queen-rbee")
    ///
    /// # Returns
    /// Path to the binary if found
    ///
    /// # Example
    /// ```rust,no_run
    /// use daemon_lifecycle::DaemonManager;
    ///
    /// # fn example() -> anyhow::Result<()> {
    /// let binary = DaemonManager::find_binary("queen-rbee")?;
    /// println!("Found at: {}", binary.display());
    /// # Ok(())
    /// # }
    /// ```
    pub fn find_binary(name: &str) -> Result<PathBuf> {
        // Try installed location first
        if let Ok(home) = std::env::var("HOME") {
            let installed_path = PathBuf::from(format!("{}/.local/bin/{}", home, name));
            if installed_path.exists() {
                n!("find_binary", "Found installed binary '{}' at: {}", name, installed_path.display());
                return Ok(installed_path);
            }
        }
        
        // Fall back to development builds
        Self::find_in_target(name)
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
            if current.join("Cargo.toml").exists()
                && (current.join("xtask").exists() || current.join("bin").exists())
            {
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
            // TEAM-311: Migrated to n!() macro
            n!("find_binary", "Found binary '{}' at: {}", name, debug_path.display());
            return Ok(debug_path);
        }

        // Try release
        let release_path = workspace_root.join("target/release").join(name);
        if release_path.exists() {
            // TEAM-311: Migrated to n!() macro
            n!("find_binary", "Found binary '{}' at: {}", name, release_path.display());
            return Ok(release_path);
        }

        // TEAM-311: Migrated to n!() macro
        n!("find_binary", "Binary '{}' not found in target/debug or target/release", name);
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
