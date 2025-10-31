//! Binary installation utilities
//!
//! TEAM-338: Extracted from status.rs - reusable across install/uninstall/status
//! TEAM-358: Removed SSH code (lifecycle-local = LOCAL only)

use observability_narration_core::n;

/// TEAM-378: RULE ZERO - Consolidated binary check with mode parameter
/// Determines what locations to check for binary existence
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckMode {
    /// Check ALL locations: target/debug/, target/release/, ~/.local/bin/
    Any,
    /// Check ONLY ~/.local/bin/ (for "is this actually installed?" checks)
    InstalledOnly,
}

/// Check if binary exists locally
///
/// TEAM-358: Checks multiple locations for binary existence
/// TEAM-377: Renamed to clarify it checks ANY location, not just installed
/// TEAM-378: RULE ZERO - Added CheckMode parameter, deleted check_binary_actually_installed()
///
/// # Arguments
/// - `daemon_name`: Binary name (e.g., "queen-rbee", "rbee-hive")
/// - `mode`: CheckMode::Any (all locations) or CheckMode::InstalledOnly (~/.local/bin/)
///
/// # Returns
/// - `true` if binary exists in checked location(s)
/// - `false` if binary doesn't exist or check fails
///
/// # Search Order (CheckMode::Any)
/// 1. target/debug/{daemon} (development builds)
/// 2. target/release/{daemon} (release builds)
/// 3. ~/.local/bin/{daemon} (installed binaries)
///
/// # Search Order (CheckMode::InstalledOnly)
/// 1. ~/.local/bin/{daemon} (installed binaries only)
pub async fn check_binary_exists(daemon_name: &str, mode: CheckMode) -> bool {
    // TEAM-378: Narrate binary check with mode context
    match mode {
        CheckMode::Any => n!("check_binary", "🔍 Checking if {} exists (any location)", daemon_name),
        CheckMode::InstalledOnly => n!("check_binary", "🔍 Checking if {} is installed in ~/.local/bin/", daemon_name),
    }

    use lifecycle_shared::BINARY_INSTALL_DIR;
    use std::path::PathBuf;

    // TEAM-378: Phase 3 - When checking ANY location, prefer production binary if installed
    if mode == CheckMode::Any {
        // Step 1: Check if ~/.local/bin exists AND is release mode (production install)
        if let Ok(home) = std::env::var("HOME") {
            let installed_path = PathBuf::from(&home).join(BINARY_INSTALL_DIR).join(daemon_name);
            
            if installed_path.exists() {
                // Check if it's release mode
                match get_binary_mode(&installed_path) {
                    Ok(build_mode) if build_mode == "release" => {
                        n!("binary_found_prod", "✅ Using production binary: {}", installed_path.display());
                        return true;
                    }
                    Ok(build_mode) => {
                        n!("binary_found_wrong_mode", "⚠️  Found {} in ~/.local/bin but it's {} mode, not release", daemon_name, build_mode);
                        // Fall through to check dev builds
                    }
                    Err(_) => {
                        // Can't determine mode, fall through to check dev builds
                    }
                }
            }
        }
        
        // Step 2: Check dev builds (fallback)
        let debug_path = PathBuf::from(format!("target/debug/{}", daemon_name));
        if debug_path.exists() {
            n!("binary_found_dev", "✅ {} found at {}", daemon_name, debug_path.display());
            return true;
        }
        
        let release_path = PathBuf::from(format!("target/release/{}", daemon_name));
        if release_path.exists() {
            n!("binary_found_release_fallback", "✅ {} found at {}", daemon_name, release_path.display());
            return true;
        }
        
        n!("binary_not_found", "❌ {} not found in any location", daemon_name);
        return false;
    }

    // InstalledOnly mode - check only ~/.local/bin/
    let home = match std::env::var("HOME") {
        Ok(h) => h,
        Err(_) => {
            n!("check_binary_no_home", "⚠️  HOME env var not set, skipping ~/{} check", BINARY_INSTALL_DIR);
            return false;
        }
    };
    
    let installed_path = PathBuf::from(home).join(BINARY_INSTALL_DIR).join(daemon_name);
    let is_installed = installed_path.exists();

    if is_installed {
        n!("check_binary_found", "✅ {} is installed at {}", daemon_name, installed_path.display());
    } else {
        n!("check_binary_not_found", "❌ {} is not installed", daemon_name);
    }

    is_installed
}

// TEAM-378: RULE ZERO - Deleted check_binary_actually_installed()
// Use check_binary_exists(daemon_name, CheckMode::InstalledOnly) instead

/// Get build mode of a binary by executing it with --build-info
///
/// TEAM-378: Phase 2 - Binary mode detection for smart binary selection
///
/// # Arguments
/// * `binary_path` - Path to the binary to check
///
/// # Returns
/// * `Ok("debug")` - Binary was built in debug mode
/// * `Ok("release")` - Binary was built in release mode
/// * `Err(_)` - Binary doesn't support --build-info or execution failed
///
/// # Example
/// ```rust,no_run
/// use std::path::PathBuf;
/// use lifecycle_local::utils::binary::get_binary_mode;
///
/// # fn example() -> anyhow::Result<()> {
/// let path = PathBuf::from("target/release/queen-rbee");
/// let mode = get_binary_mode(&path)?;
/// assert_eq!(mode, "release");
/// # Ok(())
/// # }
/// ```
pub fn get_binary_mode(binary_path: &std::path::Path) -> anyhow::Result<String> {
    use anyhow::Context;
    use std::process::Command;

    // Execute binary with --build-info flag
    let output = Command::new(binary_path)
        .arg("--build-info")
        .output()
        .with_context(|| format!("Failed to execute {} --build-info", binary_path.display()))?;

    // Check if command succeeded
    if !output.status.success() {
        anyhow::bail!(
            "Binary {} does not support --build-info flag",
            binary_path.display()
        );
    }

    // Parse output
    let mode = String::from_utf8_lossy(&output.stdout).trim().to_string();

    // Validate mode
    if mode != "debug" && mode != "release" {
        anyhow::bail!(
            "Invalid build mode '{}' from binary {}",
            mode,
            binary_path.display()
        );
    }

    Ok(mode)
}

/// Check if a binary is a release build
///
/// TEAM-378: Phase 2 - Helper function for release binary detection
///
/// # Arguments
/// * `binary_path` - Path to the binary to check
///
/// # Returns
/// * `Ok(true)` - Binary is a release build
/// * `Ok(false)` - Binary is a debug build
/// * `Err(_)` - Could not determine mode
pub fn is_release_binary(binary_path: &std::path::Path) -> anyhow::Result<bool> {
    let mode = get_binary_mode(binary_path)?;
    Ok(mode == "release")
}
