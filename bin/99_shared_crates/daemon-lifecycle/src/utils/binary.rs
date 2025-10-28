//! Binary installation utilities
//!
//! TEAM-338: Extracted from status.rs - reusable across install/uninstall/status

use crate::utils::ssh::ssh_exec;
use crate::SshConfig;
use observability_narration_core::n;

/// Check if daemon binary is installed
///
/// TEAM-338: Reusable utility for install/uninstall/status operations
/// - Localhost: Check filesystem directly (~/.local/bin/{daemon_name})
/// - Remote: SSH command (test -f ~/.local/bin/{daemon_name})
///
/// # Arguments
/// - `daemon_name`: Binary name (e.g., "queen-rbee", "rbee-hive")
/// - `ssh_config`: SSH config (use SshConfig::localhost() for local)
///
/// # Returns
/// - `true` if binary exists in ~/.local/bin/
/// - `false` if binary doesn't exist or check fails
pub async fn check_binary_installed(daemon_name: &str, ssh_config: &SshConfig) -> bool {
    // TEAM-340: Narrate binary check for visibility
    n!("check_binary", "🔍 Checking if {} is installed", daemon_name);

    let is_installed = if ssh_config.is_localhost() {
        // Localhost: Direct filesystem check
        let home = match std::env::var("HOME") {
            Ok(h) => h,
            Err(_) => {
                n!("check_binary_no_home", "⚠️  HOME env var not set, assuming not installed");
                return false;
            }
        };
        let binary_path = std::path::PathBuf::from(home).join(".local/bin").join(daemon_name);
        binary_path.exists()
    } else {
        // Remote: SSH check
        let check_cmd = format!("test -f ~/.local/bin/{} && echo 'EXISTS'", daemon_name);
        match ssh_exec(ssh_config, &check_cmd).await {
            Ok(output) => output.trim().contains("EXISTS"),
            Err(_) => false,
        }
    };

    if is_installed {
        n!("check_binary_found", "✅ {} is installed", daemon_name);
    } else {
        n!("check_binary_not_found", "❌ {} is not installed", daemon_name);
    }

    is_installed
}
