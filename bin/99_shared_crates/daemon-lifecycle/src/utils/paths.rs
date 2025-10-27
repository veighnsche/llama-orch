//! Path constants and helpers for daemon lifecycle
//!
//! TEAM-328: Centralized path logic to ensure install/uninstall/start/stop all use SAME directories
//!
//! **CRITICAL:** All daemon lifecycle operations MUST use these functions.
//! DO NOT hardcode paths anywhere else!

use anyhow::{Context, Result};
use std::path::PathBuf;

/// Get the install directory for daemon binaries
///
/// Default: ~/.local/bin
///
/// # Returns
/// * `Ok(PathBuf)` - Path to install directory
/// * `Err` - HOME environment variable not set
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::paths::get_install_dir;
///
/// # fn example() -> anyhow::Result<()> {
/// let install_dir = get_install_dir()?;
/// // Returns: /home/user/.local/bin
/// # Ok(())
/// # }
/// ```
pub fn get_install_dir() -> Result<PathBuf> {
    let home = std::env::var("HOME").context("HOME environment variable not set")?;
    Ok(PathBuf::from(format!("{}/.local/bin", home)))
}

/// Get the full install path for a daemon binary
///
/// # Arguments
/// * `daemon_name` - Name of the daemon (e.g., "queen-rbee")
///
/// # Returns
/// * `Ok(PathBuf)` - Full path to installed binary
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::paths::get_install_path;
///
/// # fn example() -> anyhow::Result<()> {
/// let path = get_install_path("queen-rbee")?;
/// // Returns: /home/user/.local/bin/queen-rbee
/// # Ok(())
/// # }
/// ```
pub fn get_install_path(daemon_name: &str) -> Result<PathBuf> {
    Ok(get_install_dir()?.join(daemon_name))
}

/// Get the PID file directory
///
/// Default: ~/.local/var/run
///
/// # Returns
/// * `Ok(PathBuf)` - Path to PID file directory
/// * `Err` - HOME environment variable not set
pub fn get_pid_dir() -> Result<PathBuf> {
    let home = std::env::var("HOME").context("HOME environment variable not set")?;
    Ok(PathBuf::from(format!("{}/.local/var/run", home)))
}

// TEAM-329: get_pid_file_path moved to utils/pid.rs (centralized PID operations)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paths_consistency() {
        // CRITICAL: Ensure install and PID paths use same HOME
        std::env::set_var("HOME", "/test/home");
        
        let install_dir = get_install_dir().unwrap();
        let pid_dir = get_pid_dir().unwrap();
        
        assert_eq!(install_dir.to_str().unwrap(), "/test/home/.local/bin");
        assert_eq!(pid_dir.to_str().unwrap(), "/test/home/.local/var/run");
        
        // Ensure full paths work
        let install_path = get_install_path("test-daemon").unwrap();
        
        assert_eq!(install_path.to_str().unwrap(), "/test/home/.local/bin/test-daemon");
        
        // TEAM-329: get_pid_file_path test moved to utils/pid.rs
    }
}
