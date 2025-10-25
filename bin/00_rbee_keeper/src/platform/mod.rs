//! Platform-specific abstractions for rbee-keeper
//!
//! TEAM-293: Created for cross-platform support (Linux, Windows, macOS)
//!
//! This module provides platform-specific implementations for operations
//! that differ across operating systems:
//! - File paths and directories
//! - Process management
//! - Binary execution
//! - SSH/remote operations

#[cfg(target_os = "linux")]
mod linux;

#[cfg(target_os = "macos")]
mod macos;

#[cfg(target_os = "windows")]
mod windows;

// Re-export platform-specific implementations
#[cfg(target_os = "linux")]
pub use linux::*;

#[cfg(target_os = "macos")]
pub use macos::*;

#[cfg(target_os = "windows")]
pub use windows::*;

use std::path::PathBuf;
use anyhow::Result;

/// Platform-specific paths and directories
pub trait PlatformPaths {
    /// Get the default configuration directory
    /// - Linux: ~/.config/rbee
    /// - macOS: ~/Library/Application Support/rbee
    /// - Windows: %APPDATA%/rbee
    fn config_dir() -> Result<PathBuf>;

    /// Get the default data directory for databases, logs, etc.
    /// - Linux: ~/.local/share/rbee
    /// - macOS: ~/Library/Application Support/rbee
    /// - Windows: %LOCALAPPDATA%/rbee
    fn data_dir() -> Result<PathBuf>;

    /// Get the default binary installation directory
    /// - Linux: ~/.local/bin
    /// - macOS: /usr/local/bin
    /// - Windows: %LOCALAPPDATA%/Programs/rbee
    fn bin_dir() -> Result<PathBuf>;

    /// Get the platform-specific executable extension
    /// - Linux: "" (no extension)
    /// - macOS: "" (no extension)
    /// - Windows: ".exe"
    fn exe_extension() -> &'static str;
}

/// Platform-specific process operations
pub trait PlatformProcess {
    /// Check if a process is running
    fn is_running(pid: u32) -> bool;

    /// Gracefully terminate a process
    fn terminate(pid: u32) -> Result<()>;

    /// Force kill a process
    fn kill(pid: u32) -> Result<()>;
}

/// Platform-specific remote operations
/// 
/// Note: SSH is available on all platforms but with different setups:
/// - Linux: Native OpenSSH
/// - macOS: Native OpenSSH
/// - Windows: OpenSSH via Windows 10+ or WSL
pub trait PlatformRemote {
    /// Check if SSH is available on this platform
    fn has_ssh_support() -> bool;

    /// Get the SSH executable name
    /// - Linux/macOS: "ssh"
    /// - Windows: "ssh.exe" (if OpenSSH installed) or requires WSL
    fn ssh_executable() -> &'static str;

    /// Check if SSH is properly configured
    fn check_ssh_available() -> Result<bool>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_paths_exist() {
        // Should not panic on any platform
        let config = config_dir();
        let data = data_dir();
        let bin = bin_dir();

        assert!(config.is_ok(), "Config dir should be accessible");
        assert!(data.is_ok(), "Data dir should be accessible");
        assert!(bin.is_ok(), "Bin dir should be accessible");
    }

    #[test]
    fn test_exe_extension() {
        let ext = exe_extension();
        
        #[cfg(target_os = "windows")]
        assert_eq!(ext, ".exe");
        
        #[cfg(not(target_os = "windows"))]
        assert_eq!(ext, "");
    }

    #[test]
    fn test_ssh_support() {
        // All platforms should report whether they support SSH
        let has_support = has_ssh_support();
        let executable = ssh_executable();
        
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        assert!(has_support, "Linux/macOS should have SSH support");
        
        // Don't assert on Windows as it depends on installation
        println!("Platform SSH support: {}", has_support);
        println!("SSH executable: {}", executable);
    }
}
