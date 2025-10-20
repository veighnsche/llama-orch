// TEAM-135: Created by TEAM-135 (scaffolding)
// TEAM-152: Implemented core daemon spawning functionality
// TEAM-152: Replaced tracing with narration for observability
// Purpose: Shared daemon lifecycle management for rbee-keeper, queen-rbee, and rbee-hive

#![warn(missing_docs)]
#![warn(clippy::all)]

//! daemon-lifecycle
//!
//! Shared daemon lifecycle management functionality for managing daemon processes
//! across rbee-keeper, queen-rbee, and rbee-hive binaries.
//! All observability is handled through narration-core (no tracing).

use anyhow::{Context, Result};
use observability_narration_core::Narration;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use tokio::process::{Child, Command};

// Actor and action constants
// TEAM-155: Added emoji prefix for visual identification
const ACTOR_DAEMON_LIFECYCLE: &str = "⚙️ daemon-lifecycle";
const ACTION_SPAWN: &str = "spawn";
const ACTION_FIND_BINARY: &str = "find_binary";

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
        Narration::new(
            ACTOR_DAEMON_LIFECYCLE,
            ACTION_SPAWN,
            self.binary_path.display().to_string(),
        )
        .human(format!(
            "Spawning daemon: {} with args: {:?}",
            self.binary_path.display(),
            self.args
        ))
        .emit();

        let child = Command::new(&self.binary_path)
            .args(&self.args)
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()
            .context(format!("Failed to spawn daemon: {}", self.binary_path.display()))?;

        let pid_str = child.id().map(|p| p.to_string()).unwrap_or_else(|| "unknown".to_string());
        Narration::new(ACTOR_DAEMON_LIFECYCLE, ACTION_SPAWN, &pid_str)
            .human(format!("Daemon spawned with PID: {}", pid_str))
            .emit();
        Ok(child)
    }

    /// Find a binary in the target directory (for development)
    ///
    /// Searches for the binary in:
    /// - `target/debug/{name}`
    /// - `target/release/{name}`
    ///
    /// # Arguments
    /// * `name` - Binary name (e.g., "queen-rbee")
    ///
    /// # Returns
    /// Path to the binary if found
    pub fn find_in_target(name: &str) -> Result<PathBuf> {
        // Try debug first (development mode)
        let debug_path = PathBuf::from("target/debug").join(name);
        if debug_path.exists() {
            Narration::new(ACTOR_DAEMON_LIFECYCLE, ACTION_FIND_BINARY, name)
                .human(format!("Found binary at: {}", debug_path.display()))
                .emit();
            return Ok(debug_path);
        }

        // Try release
        let release_path = PathBuf::from("target/release").join(name);
        if release_path.exists() {
            Narration::new(ACTOR_DAEMON_LIFECYCLE, ACTION_FIND_BINARY, name)
                .human(format!("Found binary at: {}", release_path.display()))
                .emit();
            return Ok(release_path);
        }

        Narration::new(ACTOR_DAEMON_LIFECYCLE, ACTION_FIND_BINARY, name)
            .human(format!("Binary '{}' not found in target/debug or target/release", name))
            .error_kind("binary_not_found")
            .emit();
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
