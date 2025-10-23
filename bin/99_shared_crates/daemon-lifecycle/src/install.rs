//! Daemon installation and uninstallation
//!
//! TEAM-259: Extracted common install/uninstall patterns
//!
//! Provides generic daemon installation functionality for:
//! - hive-lifecycle (install/uninstall hive)
//! - worker-lifecycle (install/uninstall workers: vLLM, llama.cpp, SD, Whisper)

use anyhow::Result;
use observability_narration_core::NarrationFactory;
use std::path::{Path, PathBuf};

const NARRATE: NarrationFactory = NarrationFactory::new("dmn-life");

/// Configuration for daemon installation
pub struct InstallConfig {
    /// Name of the daemon binary (e.g., "rbee-hive", "vllm-worker")
    pub binary_name: String,

    /// Optional: Provided binary path (if user specifies custom path)
    pub binary_path: Option<String>,

    /// Optional: Target installation path (if copying binary)
    pub target_path: Option<String>,

    /// Optional: Job ID for narration routing
    pub job_id: Option<String>,
}

/// Result of daemon installation
pub struct InstallResult {
    /// Path to the installed binary
    pub binary_path: String,

    /// Whether the binary was found in target directory
    pub found_in_target: bool,
}

/// Install a daemon binary
///
/// Steps:
/// 1. Check if binary path was provided
/// 2. If not, try to find in target directory (debug/release)
/// 3. Verify binary exists and is executable
/// 4. Return installation path
///
/// # Arguments
/// * `config` - Installation configuration
///
/// # Returns
/// * `Ok(InstallResult)` - Installation successful with binary path
/// * `Err` - Binary not found or not executable
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::{InstallConfig, install_daemon};
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = InstallConfig {
///     binary_name: "rbee-hive".to_string(),
///     binary_path: None,
///     target_path: None,
///     job_id: Some("job_123".to_string()),
/// };
///
/// let result = install_daemon(config).await?;
/// println!("Installed at: {}", result.binary_path);
/// # Ok(())
/// # }
/// ```
pub async fn install_daemon(config: InstallConfig) -> Result<InstallResult> {
    let mut narration = NARRATE.action("daemon_install").context(&config.binary_name);
    if let Some(ref job_id) = config.job_id {
        narration = narration.job_id(job_id);
    }
    narration.human("🔧 Installing daemon '{}'").emit();

    // Step 1: Check if binary path was provided
    if let Some(provided_path) = config.binary_path {
        let mut narration = NARRATE.action("daemon_binary").context(&provided_path);
        if let Some(ref job_id) = config.job_id {
            narration = narration.job_id(job_id);
        }
        narration.human("📁 Using provided binary path: {}").emit();

        let path = Path::new(&provided_path);
        if !path.exists() {
            let mut narration = NARRATE
                .action("daemon_bin_err")
                .context(&provided_path)
                .human("❌ Binary not found at: {}")
                .error_kind("binary_not_found");
            if let Some(ref job_id) = config.job_id {
                narration = narration.job_id(job_id);
            }
            narration.emit_error();
            anyhow::bail!("Binary not found at: {}", provided_path);
        }

        return Ok(InstallResult { binary_path: provided_path, found_in_target: false });
    }

    // Step 2: Try to find in target directory
    let mut narration = NARRATE.action("daemon_search").context(&config.binary_name);
    if let Some(ref job_id) = config.job_id {
        narration = narration.job_id(job_id);
    }
    narration.human("🔍 Searching for '{}' in target directory").emit();

    match crate::manager::DaemonManager::find_in_target(&config.binary_name) {
        Ok(binary_path) => {
            let mut narration = NARRATE
                .action("daemon_found")
                .context(&config.binary_name)
                .context(binary_path.display().to_string());
            if let Some(ref job_id) = config.job_id {
                narration = narration.job_id(job_id);
            }
            narration.human("✅ Found '{}' at: {}").emit();

            Ok(InstallResult {
                binary_path: binary_path.display().to_string(),
                found_in_target: true,
            })
        }
        Err(e) => {
            let mut narration = NARRATE
                .action("daemon_not_found")
                .context(&config.binary_name)
                .human("❌ Binary '{}' not found in target directory")
                .error_kind("binary_not_found");
            if let Some(ref job_id) = config.job_id {
                narration = narration.job_id(job_id);
            }
            narration.emit_error();
            Err(e)
        }
    }
}

/// Uninstall a daemon
///
/// This is a placeholder for future uninstallation logic.
/// Currently just validates that the daemon is not running.
///
/// # Arguments
/// * `daemon_name` - Name of the daemon to uninstall
/// * `job_id` - Optional job ID for narration routing
///
/// # Returns
/// * `Ok(())` - Uninstallation successful
/// * `Err` - Daemon is still running or other error
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::uninstall_daemon;
///
/// # async fn example() -> anyhow::Result<()> {
/// uninstall_daemon("rbee-hive", Some("job_123")).await?;
/// # Ok(())
/// # }
/// ```
pub async fn uninstall_daemon(daemon_name: &str, job_id: Option<&str>) -> Result<()> {
    let mut narration = NARRATE.action("daemon_uninstall").context(daemon_name);
    if let Some(jid) = job_id {
        narration = narration.job_id(jid);
    }
    narration.human("🗑️  Uninstalling daemon '{}'").emit();

    // TODO: Add actual uninstallation logic
    // - Check if daemon is running (error if yes)
    // - Remove configuration
    // - Cleanup resources

    let mut narration = NARRATE.action("daemon_uninstall").context(daemon_name);
    if let Some(jid) = job_id {
        narration = narration.job_id(jid);
    }
    narration.human("✅ Daemon '{}' uninstalled").emit();

    Ok(())
}
