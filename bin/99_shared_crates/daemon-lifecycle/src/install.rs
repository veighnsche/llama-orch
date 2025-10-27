//! Daemon installation
//!
//! TEAM-328: Provides install operations
//! TEAM-329: Moved build_daemon() to build.rs module
//!
//! Core operations:
//! - `install_daemon()` - Move binary to ~/.local/bin

use anyhow::{Context, Result};
use observability_narration_core::n;
use std::path::Path;

// TEAM-316: Use install types from types module
// TEAM-329: types/install.rs (PARITY)
pub use crate::types::install::{InstallConfig, InstallResult};

// TEAM-329: Import build_daemon from build module
use crate::build::build_daemon;

/// Install a binary to a directory (default: ~/.local/bin)
///
/// TEAM-321: Common pattern extracted from queen-lifecycle and hive-lifecycle
/// TEAM-329: Updated to use find_binary() and build_daemon() from dedicated modules
///
/// Steps:
/// 1. Find binary (using find_binary() or build if not found)
/// 2. Create install directory
/// 3. Copy binary to install_dir/{binary_name}
/// 4. Make executable (Unix)
/// 5. Verify installation
///
/// # Arguments
/// * `binary_name` - Name of the binary (e.g., "queen-rbee", "rbee-hive")
/// * `source_path` - Optional source path (if None, uses find_binary() or builds from source)
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
pub async fn install_daemon(
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
        // TEAM-328: Try to find existing binary, if not found call build_daemon()
        // TEAM-329: Use standalone find_binary function from utils/find module
        match crate::utils::find::find_binary(binary_name) {
            Ok(path) => {
                n!("found_existing", "📦 Found existing binary: {}", path.display());
                path
            }
            Err(_) => {
                n!("binary_not_found", "⚠️  Binary not found, building from source...");
                
                // TEAM-328: Use build_daemon() instead of duplicating build logic
                let binary_path = build_daemon(binary_name).await?;
                std::path::PathBuf::from(binary_path)
            }
        }
    };

    // Determine install location
    // TEAM-328: Use centralized path function to ensure consistency with uninstall
    let install_dir = if let Some(dir) = install_dir {
        std::path::PathBuf::from(dir)
    } else {
        crate::utils::paths::get_install_dir()?
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
