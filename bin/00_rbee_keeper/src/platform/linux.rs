//! Linux-specific implementations
//!
//! TEAM-293: Platform abstraction for Linux

use super::{PlatformPaths, PlatformProcess, PlatformRemote};
use anyhow::{Result, Context};
use std::path::PathBuf;

pub struct LinuxPlatform;

impl PlatformPaths for LinuxPlatform {
    fn config_dir() -> Result<PathBuf> {
        dirs::config_dir()
            .map(|p| p.join("rbee"))
            .context("Failed to get config directory")
    }

    fn data_dir() -> Result<PathBuf> {
        dirs::data_local_dir()
            .map(|p| p.join("rbee"))
            .context("Failed to get data directory")
    }

    fn bin_dir() -> Result<PathBuf> {
        dirs::home_dir()
            .map(|p| p.join(".local/bin"))
            .context("Failed to get bin directory")
    }

    fn exe_extension() -> &'static str {
        ""
    }
}

impl PlatformProcess for LinuxPlatform {
    fn is_running(pid: u32) -> bool {
        std::path::Path::new(&format!("/proc/{}", pid)).exists()
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

impl PlatformRemote for LinuxPlatform {
    fn has_ssh_support() -> bool {
        true // Linux always has SSH support
    }

    fn ssh_executable() -> &'static str {
        "ssh"
    }

    fn check_ssh_available() -> Result<bool> {
        use std::process::Command;
        
        let output = Command::new("which")
            .arg("ssh")
            .output()
            .context("Failed to check for ssh")?;
        
        Ok(output.status.success())
    }
}

// Re-export implementations
pub use LinuxPlatform as Platform;

pub fn config_dir() -> Result<PathBuf> {
    LinuxPlatform::config_dir()
}

pub fn data_dir() -> Result<PathBuf> {
    LinuxPlatform::data_dir()
}

pub fn bin_dir() -> Result<PathBuf> {
    LinuxPlatform::bin_dir()
}

pub fn exe_extension() -> &'static str {
    LinuxPlatform::exe_extension()
}

pub fn is_running(pid: u32) -> bool {
    LinuxPlatform::is_running(pid)
}

pub fn terminate(pid: u32) -> Result<()> {
    LinuxPlatform::terminate(pid)
}

pub fn kill(pid: u32) -> Result<()> {
    LinuxPlatform::kill(pid)
}

pub fn has_ssh_support() -> bool {
    LinuxPlatform::has_ssh_support()
}

pub fn ssh_executable() -> &'static str {
    LinuxPlatform::ssh_executable()
}

pub fn check_ssh_available() -> Result<bool> {
    LinuxPlatform::check_ssh_available()
}
