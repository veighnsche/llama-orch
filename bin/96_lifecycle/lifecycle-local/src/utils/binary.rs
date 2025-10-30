//! Binary installation utilities
//!
//! TEAM-338: Extracted from status.rs - reusable across install/uninstall/status
//! TEAM-358: Removed SSH code (lifecycle-local = LOCAL only)

use observability_narration_core::n;

/// Check if daemon binary is installed locally
///
/// TEAM-358: LOCAL-only version (no SSH)
///
/// # Arguments
/// - `daemon_name`: Binary name (e.g., "queen-rbee", "rbee-hive")
///
/// # Returns
/// - `true` if binary exists in ~/.local/bin/ or target/
/// - `false` if binary doesn't exist or check fails
pub async fn check_binary_installed(daemon_name: &str) -> bool {
    // TEAM-358: Narrate binary check for visibility
    n!("check_binary", "🔍 Checking if {} is installed locally", daemon_name);

    // Check local filesystem
    let home = match std::env::var("HOME") {
        Ok(h) => h,
        Err(_) => {
            n!("check_binary_no_home", "⚠️  HOME env var not set, assuming not installed");
            return false;
        }
    };
    
    let binary_path = std::path::PathBuf::from(home).join(".local/bin").join(daemon_name);
    let is_installed = binary_path.exists();

    if is_installed {
        n!("check_binary_found", "✅ {} is installed at {}", daemon_name, binary_path.display());
    } else {
        n!("check_binary_not_found", "❌ {} is not installed", daemon_name);
    }

    is_installed
}
