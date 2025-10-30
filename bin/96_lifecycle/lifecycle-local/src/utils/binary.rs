//! Binary installation utilities
//!
//! TEAM-338: Extracted from status.rs - reusable across install/uninstall/status
//! TEAM-358: Removed SSH code (lifecycle-local = LOCAL only)

use observability_narration_core::n;

/// Check if daemon binary is installed locally
///
/// TEAM-358: LOCAL-only version (no SSH)
/// TEAM-367: Fixed to check target/ directories (for development builds)
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
pub async fn check_binary_installed(daemon_name: &str) -> bool {
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
    let home = match std::env::var("HOME") {
        Ok(h) => h,
        Err(_) => {
            n!("check_binary_no_home", "⚠️  HOME env var not set, assuming not installed");
            return false;
        }
    };
    
    let installed_path = std::path::PathBuf::from(home).join(".local/bin").join(daemon_name);
    let is_installed = installed_path.exists();

    if is_installed {
        n!("check_binary_found", "✅ {} is installed at {}", daemon_name, installed_path.display());
    } else {
        n!("check_binary_not_found", "❌ {} is not installed", daemon_name);
    }

    is_installed
}
