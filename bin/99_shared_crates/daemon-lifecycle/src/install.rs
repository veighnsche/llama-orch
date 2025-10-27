//! Daemon installation
//!
//! TEAM-323: DELETED install_daemon() - RULE ZERO violation (duplicate of install_to_local_bin)
//!
//! Provides daemon installation functionality.
//! Use `install_to_local_bin()` for all installations.

use anyhow::{Context, Result};
use observability_narration_core::n;
use std::path::Path;

// TEAM-316: Use install types from daemon-contract
// TEAM-323: InstallConfig/InstallResult only used by deleted install_daemon()
pub use daemon_contract::{InstallConfig, InstallResult, UninstallConfig};

/// Install a binary to a directory (default: ~/.local/bin)
///
/// TEAM-321: Common pattern extracted from queen-lifecycle and hive-lifecycle
///
/// Steps:
/// 1. Find binary (using DaemonManager::find_binary)
/// 2. Create install directory
/// 3. Copy binary to install_dir/{binary_name}
/// 4. Make executable (Unix)
/// 5. Verify installation
///
/// # Arguments
/// * `binary_name` - Name of the binary (e.g., "queen-rbee", "rbee-hive")
/// * `source_path` - Optional source path (if None, uses DaemonManager::find_binary)
/// * `install_dir` - Optional install directory (if None, uses ~/.local/bin)
///
/// # Returns
/// * `Ok(String)` - Installation path
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::install_to_local_bin;
///
/// # async fn example() -> anyhow::Result<()> {
/// // Install to default ~/.local/bin
/// let path = install_to_local_bin("queen-rbee", None, None).await?;
///
/// // Install to custom directory
/// let path = install_to_local_bin("rbee-hive", None, Some("/opt/bin".to_string())).await?;
/// # Ok(())
/// # }
/// ```
pub async fn install_to_local_bin(
    binary_name: &str,
    source_path: Option<String>,
    install_dir: Option<String>,
) -> Result<String> {
    let dir_display = install_dir.as_deref().unwrap_or("~/.local/bin");
    n!("install_to_local_bin", "📦 Installing {} to {}", binary_name, dir_display);

    // Find source binary
    let source = if let Some(path) = source_path {
        let p = Path::new(&path);
        if !p.exists() {
            anyhow::bail!("Binary not found at: {}", path);
        }
        std::path::PathBuf::from(path)
    } else {
        crate::manager::DaemonManager::find_binary(binary_name)?
    };

    // Determine install location
    let install_dir = if let Some(dir) = install_dir {
        std::path::PathBuf::from(dir)
    } else {
        let home = std::env::var("HOME").context("HOME environment variable not set")?;
        std::path::PathBuf::from(format!("{}/.local/bin", home))
    };
    let install_path = install_dir.join(binary_name);

    // Check if already installed
    if install_path.exists() {
        anyhow::bail!(
            "{} already installed at {}. Uninstall first or use rebuild.",
            binary_name,
            install_path.display()
        );
    }

    // Create install directory
    std::fs::create_dir_all(&install_dir)
        .context("Failed to create ~/.local/bin directory")?;

    // Copy binary
    n!("copy_binary", "📋 Copying to {}", install_path.display());
    std::fs::copy(&source, &install_path)
        .context(format!("Failed to copy {} to {}", binary_name, install_path.display()))?;

    // Make executable (Unix only)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&install_path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&install_path, perms)?;
    }

    // Verify installation
    let output = std::process::Command::new(&install_path)
        .arg("--version")
        .output();

    let version = if let Ok(out) = output {
        String::from_utf8_lossy(&out.stdout).trim().to_string()
    } else {
        "unknown".to_string()
    };

    n!("install_complete", "✅ {} installed successfully!", binary_name);
    n!("install_path", "📍 Binary location: {}", install_path.display());
    n!("install_version", "📦 Version: {}", version);
    n!("install_info", "💡 Make sure ~/.local/bin is in your PATH");

    Ok(install_path.display().to_string())
}