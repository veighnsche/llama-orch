//! macOS-specific implementations
//!
//! TEAM-293: Platform abstraction for macOS

use super::{PlatformPaths, PlatformProcess, PlatformRemote};
use anyhow::{Context, Result};
use std::path::PathBuf;

pub struct MacOSPlatform;

impl PlatformPaths for MacOSPlatform {
    fn config_dir() -> Result<PathBuf> {
        // macOS uses ~/Library/Application Support
        dirs::home_dir()
            .map(|p| p.join("Library/Application Support/rbee"))
            .context("Failed to get config directory")
    }

    fn data_dir() -> Result<PathBuf> {
        // Same as config on macOS
        Self::config_dir()
    }

    fn bin_dir() -> Result<PathBuf> {
        // macOS traditionally uses /usr/local/bin for user binaries
        Ok(PathBuf::from("/usr/local/bin"))
    }

    fn exe_extension() -> &'static str {
        ""
    }
}

impl PlatformProcess for MacOSPlatform {
    fn is_running(pid: u32) -> bool {
        use std::process::Command;

        // Use ps command on macOS
        Command::new("ps")
            .arg("-p")
            .arg(pid.to_string())
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false)
    }

    fn terminate(pid: u32) -> Result<()> {
        use std::process::Command;

        Command::new("kill")
            .arg("-TERM")
            .arg(pid.to_string())
            .output()
            .context("Failed to send SIGTERM")?;

        Ok(())
    }

    fn kill(pid: u32) -> Result<()> {
        use std::process::Command;

        Command::new("kill")
            .arg("-KILL")
            .arg(pid.to_string())
            .output()
            .context("Failed to send SIGKILL")?;

        Ok(())
    }
}

impl PlatformRemote for MacOSPlatform {
    fn has_ssh_support() -> bool {
        true // macOS has native SSH support
    }

    fn ssh_executable() -> &'static str {
        "ssh"
    }

    fn check_ssh_available() -> Result<bool> {
        use std::process::Command;

        let output =
            Command::new("which").arg("ssh").output().context("Failed to check for ssh")?;

        Ok(output.status.success())
    }
}

// Re-export implementations
pub use MacOSPlatform as Platform;

pub fn config_dir() -> Result<PathBuf> {
    MacOSPlatform::config_dir()
}

pub fn data_dir() -> Result<PathBuf> {
    MacOSPlatform::data_dir()
}

pub fn bin_dir() -> Result<PathBuf> {
    MacOSPlatform::bin_dir()
}

pub fn exe_extension() -> &'static str {
    MacOSPlatform::exe_extension()
}

pub fn is_running(pid: u32) -> bool {
    MacOSPlatform::is_running(pid)
}

pub fn terminate(pid: u32) -> Result<()> {
    MacOSPlatform::terminate(pid)
}

pub fn kill(pid: u32) -> Result<()> {
    MacOSPlatform::kill(pid)
}

pub fn has_ssh_support() -> bool {
    MacOSPlatform::has_ssh_support()
}

pub fn ssh_executable() -> &'static str {
    MacOSPlatform::ssh_executable()
}

pub fn check_ssh_available() -> Result<bool> {
    MacOSPlatform::check_ssh_available()
}
