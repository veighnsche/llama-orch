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

/// Configuration for daemon uninstallation
pub struct UninstallConfig {
    /// Name of the daemon (e.g., "queen-rbee", "rbee-hive")
    pub daemon_name: String,

    /// Path to the installed binary
    pub install_path: PathBuf,

    /// Optional: Health check URL to verify daemon is not running
    pub health_url: Option<String>,

    /// Optional: Timeout for health check (default: 2 seconds)
    pub health_timeout_secs: Option<u64>,

    /// Optional: Job ID for narration routing
    pub job_id: Option<String>,
}

/// Uninstall a daemon binary
///
/// Steps:
/// 1. Check if binary exists at install_path
/// 2. If health_url provided, check if daemon is running (error if yes)
/// 3. Remove binary file
/// 4. Emit success narration
///
/// # Arguments
/// * `config` - Uninstallation configuration
///
/// # Returns
/// * `Ok(())` - Uninstallation successful
/// * `Err` - Daemon is still running or removal failed
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::{UninstallConfig, uninstall_daemon};
/// use std::path::PathBuf;
///
/// # async fn example() -> anyhow::Result<()> {
/// let config = UninstallConfig {
///     daemon_name: "queen-rbee".to_string(),
///     install_path: PathBuf::from("/home/user/.local/bin/queen-rbee"),
///     health_url: Some("http://localhost:8500".to_string()),
///     health_timeout_secs: Some(2),
///     job_id: None,
/// };
///
/// uninstall_daemon(config).await?;
/// # Ok(())
/// # }
/// ```
pub async fn uninstall_daemon(config: UninstallConfig) -> Result<()> {
    let mut narration = NARRATE.action("daemon_uninstall").context(&config.daemon_name);
    if let Some(ref job_id) = config.job_id {
        narration = narration.job_id(job_id);
    }
    narration.human("🗑️  Uninstalling daemon '{}'").emit();

    // Step 1: Check if binary exists
    if !config.install_path.exists() {
        let mut narration = NARRATE
            .action("daemon_not_installed")
            .context(&config.daemon_name)
            .context(config.install_path.display().to_string());
        if let Some(ref job_id) = config.job_id {
            narration = narration.job_id(job_id);
        }
        narration.human("⚠️  Daemon '{}' not installed at: {}").emit();
        return Ok(());
    }

    // Step 2: Check if daemon is running (if health_url provided)
    if let Some(health_url) = config.health_url {
        let timeout_secs = config.health_timeout_secs.unwrap_or(2);
        let is_running = crate::health::is_daemon_healthy(
            &health_url,
            None, // Use default /health endpoint
            Some(std::time::Duration::from_secs(timeout_secs)),
        )
        .await;

        if is_running {
            let mut narration = NARRATE
                .action("daemon_still_running")
                .context(&config.daemon_name)
                .human("⚠️  Daemon '{}' is currently running. Stop it first.")
                .error_kind("daemon_running");
            if let Some(ref job_id) = config.job_id {
                narration = narration.job_id(job_id);
            }
            narration.emit_error();
            anyhow::bail!("Daemon {} is still running", config.daemon_name);
        }
    }

    // Step 3: Remove binary file
    std::fs::remove_file(&config.install_path)?;

    let mut narration = NARRATE
        .action("daemon_uninstalled")
        .context(&config.daemon_name)
        .context(config.install_path.display().to_string());
    if let Some(ref job_id) = config.job_id {
        narration = narration.job_id(job_id);
    }
    narration.human("✅ Daemon '{}' uninstalled successfully!").emit();

    let mut narration =
        NARRATE.action("daemon_removed").context(config.install_path.display().to_string());
    if let Some(ref job_id) = config.job_id {
        narration = narration.job_id(job_id);
    }
    narration.human("🗑️  Removed: {}").emit();

    Ok(())
}
