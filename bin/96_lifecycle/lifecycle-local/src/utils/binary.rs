//! Binary installation utilities
//!
//! TEAM-338: Extracted from status.rs - reusable across install/uninstall/status
//! TEAM-358: Removed SSH code (lifecycle-local = LOCAL only)

use observability_narration_core::n;

/// Check if binary exists locally (dev, release, or installed)
///
/// TEAM-358: Checks multiple locations for binary existence
/// TEAM-377: Renamed to clarify it checks ANY location, not just installed
///
/// # Arguments
/// - `daemon_name`: Binary name (e.g., "queen-rbee", "rbee-hive")
///
/// # Returns
/// - `true` if binary exists in target/debug/, target/release/, or ~/.local/bin/
/// - `false` if binary doesn't exist or check fails
///
/// # Search Order
/// 1. target/debug/{daemon} (development builds)
/// 2. target/release/{daemon} (release builds)
/// 3. ~/.local/bin/{daemon} (installed binaries)
pub async fn check_binary_exists(daemon_name: &str) -> bool {
    // TEAM-358: Narrate binary check for visibility
    n!("check_binary", "🔍 Checking if {} is installed locally", daemon_name);

    // TEAM-367: Check target/ directories first (for development)
    use std::path::PathBuf;
    
    let debug_path = PathBuf::from(format!("target/debug/{}", daemon_name));
    if debug_path.exists() {
        n!("check_binary_found", "✅ {} found at {}", daemon_name, debug_path.display());
        return true;
    }
    
    let release_path = PathBuf::from(format!("target/release/{}", daemon_name));
    if release_path.exists() {
        n!("check_binary_found", "✅ {} found at {}", daemon_name, release_path.display());
        return true;
    }

    // Check ~/.local/bin/ (installed binaries)
    // TEAM-377: RULE ZERO - Use constant for install directory
    use lifecycle_shared::BINARY_INSTALL_DIR;
    
    let home = match std::env::var("HOME") {
        Ok(h) => h,
        Err(_) => {
            n!("check_binary_no_home", "⚠️  HOME env var not set, skipping ~/{} check", BINARY_INSTALL_DIR);
            return false;
        }
    };
    
    let installed_path = std::path::PathBuf::from(home).join(BINARY_INSTALL_DIR).join(daemon_name);
    let is_installed = installed_path.exists();

    if is_installed {
        n!("check_binary_found", "✅ {} is installed at {}", daemon_name, installed_path.display());
    } else {
        n!("check_binary_not_found", "❌ {} is not installed", daemon_name);
    }

    is_installed
}

/// Check if binary is ACTUALLY installed (only checks ~/.local/bin/)
///
/// TEAM-377: NEW - Separate function for checking if binary is installed vs just exists
///
/// # Arguments
/// - `daemon_name`: Binary name (e.g., "queen-rbee", "rbee-hive")
///
/// # Returns
/// - `true` ONLY if binary exists in ~/.local/bin/
/// - `false` if binary doesn't exist in ~/.local/bin/
///
/// # Use Cases
/// - Enable/disable "Uninstall" button (only show for installed binaries)
/// - Distinguish between dev builds and installed binaries
///
/// # Note
/// This does NOT check target/debug/ or target/release/.
/// Use `check_binary_exists()` if you want to check all locations.
pub async fn check_binary_actually_installed(daemon_name: &str) -> bool {
    use lifecycle_shared::BINARY_INSTALL_DIR;
    
    let home = match std::env::var("HOME") {
        Ok(h) => h,
        Err(_) => {
            n!("check_installed_no_home", "⚠️  HOME env var not set");
            return false;
        }
    };
    
    let installed_path = std::path::PathBuf::from(home).join(BINARY_INSTALL_DIR).join(daemon_name);
    installed_path.exists()
}
