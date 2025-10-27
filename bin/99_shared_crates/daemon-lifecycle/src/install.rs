//! Daemon installation and uninstallation
//!
//! TEAM-259: Extracted common install/uninstall patterns
//! TEAM-316: Use types from daemon-contract
//!
//! Provides generic daemon installation functionality for:
//! - hive-lifecycle (install/uninstall hive)
//! - worker-lifecycle (install/uninstall workers: vLLM, llama.cpp, SD, Whisper)

use anyhow::Result;
use observability_narration_core::{n, with_narration_context, NarrationContext};
use std::path::Path;

// TEAM-316: Use install types from daemon-contract
pub use daemon_contract::{InstallConfig, InstallResult, UninstallConfig};

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
    // TEAM-311: Migrated to n!() macro
    let ctx = config.job_id.as_ref().map(|jid| NarrationContext::new().with_job_id(jid));
    
    let install_impl = async {
        n!("daemon_install", "🔧 Installing daemon '{}'", config.binary_name);

        // Step 1: Check if binary path was provided
        if let Some(provided_path) = config.binary_path {
            n!("daemon_binary", "📁 Using provided binary path: {}", provided_path);

            let path = Path::new(&provided_path);
            if !path.exists() {
                n!("daemon_bin_err", "❌ Binary not found at: {}", provided_path);
                anyhow::bail!("Binary not found at: {}", provided_path);
            }

            return Ok(InstallResult {
                binary_path: provided_path,
                install_time: std::time::SystemTime::now(),
                found_in_target: false,
            });
        }

        // Step 2: Try to find in target directory
        n!("daemon_search", "🔍 Searching for '{}' in target directory", config.binary_name);

        match crate::manager::DaemonManager::find_in_target(&config.binary_name) {
            Ok(binary_path) => {
                n!("daemon_found", "✅ Found '{}' at: {}", config.binary_name, binary_path.display());

                Ok(InstallResult {
                    binary_path: binary_path.display().to_string(),
                    install_time: std::time::SystemTime::now(),
                    found_in_target: true,
                })
            }
            Err(_) => {
                // TEAM-290: Binary not found, try to build it
                n!("daemon_build", "🔨 Building '{}' (not found in target)", config.binary_name);

            // Build the binary
            let output = tokio::process::Command::new("cargo")
                .arg("build")
                .arg("-p")
                .arg(&config.binary_name)
                .output()
                .await?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    n!("daemon_build_err", "❌ Failed to build '{}'", config.binary_name);
                    anyhow::bail!("Failed to build '{}': {}", config.binary_name, stderr);
                }

                // Try to find again after build
                match crate::manager::DaemonManager::find_in_target(&config.binary_name) {
                    Ok(binary_path) => {
                        n!("daemon_built", "✅ Built '{}' at: {}", config.binary_name, binary_path.display());

                        Ok(InstallResult {
                            binary_path: binary_path.display().to_string(),
                            install_time: std::time::SystemTime::now(),
                            found_in_target: true,
                        })
                    }
                    Err(e) => {
                        n!("daemon_not_found", "❌ Binary '{}' not found even after build", config.binary_name);
                        Err(e)
                    }
                }
            }
        }
    };
    
    // Execute with context if job_id provided
    if let Some(ctx) = ctx {
        with_narration_context(ctx, install_impl).await
    } else {
        install_impl.await
    }
}