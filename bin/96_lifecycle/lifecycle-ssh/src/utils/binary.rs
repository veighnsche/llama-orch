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
/// TEAM-367: Fixed to check target/ directories (for development builds)
///
/// # Arguments
/// - `daemon_name`: Binary name (e.g., "queen-rbee", "rbee-hive")
/// - `ssh_config`: SSH config for remote machine
///
/// # Returns
/// - `true` if binary exists in target/debug/, target/release/, or ~/.local/bin/ on remote
/// - `false` if binary doesn't exist or check fails
///
/// # Search Order
/// 1. target/debug/{daemon} (development builds)
/// 2. target/release/{daemon} (release builds)
/// 3. ~/.local/bin/{daemon} (installed binaries)
pub async fn check_binary_installed(daemon_name: &str, ssh_config: &SshConfig) -> bool {
    // TEAM-358: Narrate binary check for visibility
    n!("check_binary", "🔍 Checking if {} is installed on {}@{}", daemon_name, ssh_config.user, ssh_config.hostname);

    // TEAM-367: Check all three locations (same as local version)
    // Check target/debug, target/release, and ~/.local/bin
    let check_cmd = format!(
        "(test -f target/debug/{} && echo 'target/debug/{}') || \
         (test -f target/release/{} && echo 'target/release/{}') || \
         (test -f ~/.local/bin/{} && echo '~/.local/bin/{}') || \
         echo 'NOT_FOUND'",
        daemon_name, daemon_name, daemon_name, daemon_name, daemon_name, daemon_name
    );
    
    let result = match ssh_exec(ssh_config, &check_cmd).await {
        Ok(output) => output.trim().to_string(),
        Err(_) => "NOT_FOUND".to_string(),
    };

    let is_installed = result != "NOT_FOUND";

    if is_installed {
        n!("check_binary_found", "✅ {} found at {}", daemon_name, result);
    } else {
        n!("check_binary_not_found", "❌ {} is not installed", daemon_name);
    }

    is_installed
}
