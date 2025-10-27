//! PID file management utilities
//!
//! TEAM-329: Centralized PID operations (extracted from start.rs and stop.rs)
//!
//! Standard Unix pattern: PID files in ~/.local/var/run/{daemon}.pid

use anyhow::{Context, Result};
use std::path::PathBuf;

use crate::utils::paths::get_pid_dir;

/// Get the full path to a daemon's PID file
///
/// # Arguments
/// * `daemon_name` - Name of the daemon (e.g., "queen-rbee", "rbee-hive")
///
/// # Returns
/// * `Ok(PathBuf)` - Full path to PID file (e.g., ~/.local/var/run/queen-rbee.pid)
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::get_pid_file_path;
///
/// # fn example() -> anyhow::Result<()> {
/// let path = get_pid_file_path("queen-rbee")?;
/// // Returns: /home/user/.local/var/run/queen-rbee.pid
/// # Ok(())
/// # }
/// ```
pub fn get_pid_file_path(daemon_name: &str) -> Result<PathBuf> {
    let run_dir = get_pid_dir()?;
    
    // Create directory if it doesn't exist
    std::fs::create_dir_all(&run_dir)?;
    
    Ok(run_dir.join(format!("{}.pid", daemon_name)))
}

/// Write PID to file
///
/// TEAM-327: Standard Unix pattern for stateless daemon management
///
/// # Arguments
/// * `daemon_name` - Name of the daemon
/// * `pid` - Process ID to write
///
/// # Returns
/// * `Ok(())` - PID file written successfully
/// * `Err` - Failed to write PID file
pub fn write_pid_file(daemon_name: &str, pid: u32) -> Result<()> {
    let pid_file = get_pid_file_path(daemon_name)?;
    std::fs::write(&pid_file, pid.to_string())
        .context(format!("Failed to write PID file: {}", pid_file.display()))?;
    Ok(())
}

/// Read PID from PID file
///
/// TEAM-327: Standard Unix pattern for stateless daemon management
///
/// # Arguments
/// * `daemon_name` - Name of the daemon
///
/// # Returns
/// * `Ok(u32)` - Process ID read from file
/// * `Err` - PID file not found or invalid
pub fn read_pid_file(daemon_name: &str) -> Result<u32> {
    let pid_file = get_pid_file_path(daemon_name)?;
    
    if !pid_file.exists() {
        anyhow::bail!("PID file not found: {}. Is {} running?", pid_file.display(), daemon_name);
    }
    
    let pid_str = std::fs::read_to_string(&pid_file)
        .context(format!("Failed to read PID file: {}", pid_file.display()))?;
    
    let pid = pid_str.trim().parse::<u32>()
        .context(format!("Invalid PID in {}: {}", pid_file.display(), pid_str))?;
    
    Ok(pid)
}

/// Remove PID file after successful shutdown
///
/// TEAM-327: Cleanup PID file to avoid stale entries
///
/// # Arguments
/// * `daemon_name` - Name of the daemon
///
/// # Returns
/// * `Ok(())` - PID file removed (or didn't exist)
/// * `Err` - Failed to remove PID file
pub fn remove_pid_file(daemon_name: &str) -> Result<()> {
    let pid_file = get_pid_file_path(daemon_name)?;
    if pid_file.exists() {
        std::fs::remove_file(&pid_file)
            .context(format!("Failed to remove PID file: {}", pid_file.display()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_pid_file_path() {
        // Test path construction without creating directories
        std::env::set_var("HOME", "/test/home");
        
        // Use get_pid_dir directly to avoid directory creation
        let pid_dir = get_pid_dir().unwrap();
        let expected_path = pid_dir.join("test-daemon.pid");
        
        assert_eq!(pid_dir.to_str().unwrap(), "/test/home/.local/var/run");
        assert_eq!(expected_path.to_str().unwrap(), "/test/home/.local/var/run/test-daemon.pid");
    }
}
