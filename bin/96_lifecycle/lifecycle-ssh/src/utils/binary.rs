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

/// Get build mode of a remote binary by executing it with --build-info via SSH
///
/// TEAM-379: Remote binary mode detection for UI display
///
/// # Arguments
/// * `daemon_name` - Binary name (e.g., "queen-rbee", "rbee-hive")
/// * `ssh_config` - SSH config for remote machine
///
/// # Returns
/// * `Ok("debug")` - Binary was built in debug mode
/// * `Ok("release")` - Binary was built in release mode
/// * `Err(_)` - Binary doesn't support --build-info or execution failed
///
/// # Example
/// ```rust,no_run
/// use lifecycle_ssh::utils::binary::get_remote_binary_mode;
/// use lifecycle_ssh::SshConfig;
///
/// # async fn example() -> anyhow::Result<()> {
/// let ssh_config = SshConfig {
///     hostname: "localhost".to_string(),
///     user: "vince".to_string(),
///     port: 22,
/// };
/// let mode = get_remote_binary_mode("queen-rbee", &ssh_config).await?;
/// assert_eq!(mode, "release");
/// # Ok(())
/// # }
/// ```
pub async fn get_remote_binary_mode(daemon_name: &str, ssh_config: &SshConfig) -> anyhow::Result<String> {
    use anyhow::Context;

    // TEAM-379: Execute binary with --build-info flag via SSH
    let check_cmd = format!("~/.local/bin/{} --build-info", daemon_name);
    
    n!("check_build_mode", "🔍 Checking build mode of {} on {}@{}", daemon_name, ssh_config.user, ssh_config.hostname);
    
    let output = ssh_exec(ssh_config, &check_cmd)
        .await
        .with_context(|| format!("Failed to execute {} --build-info via SSH", daemon_name))?;

    // Parse output (should be "debug" or "release")
    let mode = output.trim().to_string();

    // Validate mode
    if mode != "debug" && mode != "release" {
        anyhow::bail!(
            "Invalid build mode '{}' from remote binary {}",
            mode,
            daemon_name
        );
    }

    n!("build_mode_detected", "✅ {} is {} mode", daemon_name, mode);

    Ok(mode)
}

/// Check if a remote binary is a release build
///
/// TEAM-379: Helper function for release binary detection
///
/// # Arguments
/// * `daemon_name` - Binary name (e.g., "queen-rbee", "rbee-hive")
/// * `ssh_config` - SSH config for remote machine
///
/// # Returns
/// * `Ok(true)` - Binary is a release build
/// * `Ok(false)` - Binary is a debug build
/// * `Err(_)` - Could not determine mode
pub async fn is_remote_release_binary(daemon_name: &str, ssh_config: &SshConfig) -> anyhow::Result<bool> {
    let mode = get_remote_binary_mode(daemon_name, ssh_config).await?;
    Ok(mode == "release")
}
