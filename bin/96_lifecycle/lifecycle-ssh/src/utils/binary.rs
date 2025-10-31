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
/// TEAM-378: RULE ZERO - Only checks ~/.local/bin/ (we never build remotely)
///
/// # Arguments
/// - `daemon_name`: Binary name (e.g., "queen-rbee", "rbee-hive")
/// - `ssh_config`: SSH config for remote machine
///
/// # Returns
/// - `true` if binary exists in ~/.local/bin/ on remote
/// - `false` if binary doesn't exist or check fails
///
/// # Why Only ~/.local/bin/?
/// SSH workflow: Build locally → SCP to ~/.local/bin/ → Never build remotely
/// We never have target/debug or target/release on remote machines.
pub async fn check_binary_installed(daemon_name: &str, ssh_config: &SshConfig) -> bool {
    // TEAM-378: Narrate binary check for visibility
    n!("check_binary", "🔍 Checking if {} is installed in ~/.local/bin/ on {}@{}", daemon_name, ssh_config.user, ssh_config.hostname);

    // TEAM-378: RULE ZERO - Only check ~/.local/bin/ (we never build remotely)
    let check_cmd = format!("test -f ~/.local/bin/{}", daemon_name);
    
    let is_installed = match ssh_exec(ssh_config, &check_cmd).await {
        Ok(_) => true,  // Exit code 0 = file exists
        Err(_) => false, // Exit code 1 = file doesn't exist
    };

    if is_installed {
        n!("check_binary_found", "✅ {} is installed at ~/.local/bin/{}", daemon_name, daemon_name);
    } else {
        n!("check_binary_not_found", "❌ {} is not installed in ~/.local/bin/", daemon_name);
    }

    is_installed
}
