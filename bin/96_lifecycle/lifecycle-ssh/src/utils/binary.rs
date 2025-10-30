//! Binary installation utilities
//!
//! TEAM-338: Extracted from status.rs - reusable across install/uninstall/status

use crate::utils::ssh::ssh_exec;
use crate::SshConfig;
use observability_narration_core::n;

/// Check if daemon binary is installed on REMOTE machine
///
/// TEAM-338: Reusable utility for install/uninstall/status operations
/// TEAM-358: Removed localhost bypass - ALWAYS uses SSH
///
/// # Arguments
/// - `daemon_name`: Binary name (e.g., "queen-rbee", "rbee-hive")
/// - `ssh_config`: SSH config for remote machine
///
/// # Returns
/// - `true` if binary exists in ~/.local/bin/ on remote machine
/// - `false` if binary doesn't exist or check fails
pub async fn check_binary_installed(daemon_name: &str, ssh_config: &SshConfig) -> bool {
    // TEAM-358: Narrate binary check for visibility
    n!("check_binary", "🔍 Checking if {} is installed on {}@{}", daemon_name, ssh_config.user, ssh_config.hostname);

    // TEAM-358: Always use SSH, even for localhost
    let check_cmd = format!("test -f ~/.local/bin/{} && echo 'EXISTS'", daemon_name);
    let is_installed = match ssh_exec(ssh_config, &check_cmd).await {
        Ok(output) => output.trim().contains("EXISTS"),
        Err(_) => false,
    };

    if is_installed {
        n!("check_binary_found", "✅ {} is installed", daemon_name);
    } else {
        n!("check_binary_not_found", "❌ {} is not installed", daemon_name);
    }

    is_installed
}
