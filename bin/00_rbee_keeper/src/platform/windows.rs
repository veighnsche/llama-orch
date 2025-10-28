//! Windows-specific implementations
//!
//! TEAM-293: Platform abstraction for Windows

use super::{PlatformPaths, PlatformProcess, PlatformRemote};
use anyhow::{bail, Context, Result};
use std::path::PathBuf;

pub struct WindowsPlatform;

impl PlatformPaths for WindowsPlatform {
    fn config_dir() -> Result<PathBuf> {
        // Windows uses %APPDATA%
        dirs::config_dir().map(|p| p.join("rbee")).context("Failed to get config directory")
    }

    fn data_dir() -> Result<PathBuf> {
        // Windows uses %LOCALAPPDATA%
        dirs::data_local_dir().map(|p| p.join("rbee")).context("Failed to get data directory")
    }

    fn bin_dir() -> Result<PathBuf> {
        // Windows: %LOCALAPPDATA%/Programs/rbee
        dirs::data_local_dir()
            .map(|p| p.join("Programs/rbee"))
            .context("Failed to get bin directory")
    }

    fn exe_extension() -> &'static str {
        ".exe"
    }
}

impl PlatformProcess for WindowsPlatform {
    fn is_running(pid: u32) -> bool {
        use std::process::Command;

        // Use tasklist on Windows
        Command::new("tasklist")
            .arg("/FI")
            .arg(format!("PID eq {}", pid))
            .output()
            .map(|output| String::from_utf8_lossy(&output.stdout).contains(&pid.to_string()))
            .unwrap_or(false)
    }

    fn terminate(pid: u32) -> Result<()> {
        use std::process::Command;

        // Use taskkill on Windows
        let output = Command::new("taskkill")
            .arg("/PID")
            .arg(pid.to_string())
            .output()
            .context("Failed to execute taskkill")?;

        if !output.status.success() {
            bail!(
                "Failed to terminate process {}: {}",
                pid,
                String::from_utf8_lossy(&output.stderr)
            );
        }

        Ok(())
    }

    fn kill(pid: u32) -> Result<()> {
        use std::process::Command;

        // Use taskkill /F (force) on Windows
        let output = Command::new("taskkill")
            .arg("/F")
            .arg("/PID")
            .arg(pid.to_string())
            .output()
            .context("Failed to execute taskkill /F")?;

        if !output.status.success() {
            bail!("Failed to kill process {}: {}", pid, String::from_utf8_lossy(&output.stderr));
        }

        Ok(())
    }
}

impl PlatformRemote for WindowsPlatform {
    fn has_ssh_support() -> bool {
        // Windows 10+ has OpenSSH, but it's not always installed
        // Check if ssh.exe exists
        Self::check_ssh_available().unwrap_or(false)
    }

    fn ssh_executable() -> &'static str {
        "ssh.exe"
    }

    fn check_ssh_available() -> Result<bool> {
        use std::process::Command;

        // Try to run ssh --version
        let output = Command::new("ssh.exe").arg("--version").output();

        match output {
            Ok(output) => Ok(output.status.success()),
            Err(_) => {
                // Try where command to find ssh.exe
                let where_output = Command::new("where").arg("ssh.exe").output();

                Ok(where_output.map(|o| o.status.success()).unwrap_or(false))
            }
        }
    }
}

// Re-export implementations
pub use WindowsPlatform as Platform;

pub fn config_dir() -> Result<PathBuf> {
    WindowsPlatform::config_dir()
}

pub fn data_dir() -> Result<PathBuf> {
    WindowsPlatform::data_dir()
}

pub fn bin_dir() -> Result<PathBuf> {
    WindowsPlatform::bin_dir()
}

pub fn exe_extension() -> &'static str {
    WindowsPlatform::exe_extension()
}

pub fn is_running(pid: u32) -> bool {
    WindowsPlatform::is_running(pid)
}

pub fn terminate(pid: u32) -> Result<()> {
    WindowsPlatform::terminate(pid)
}

pub fn kill(pid: u32) -> Result<()> {
    WindowsPlatform::kill(pid)
}

pub fn has_ssh_support() -> bool {
    WindowsPlatform::has_ssh_support()
}

pub fn ssh_executable() -> &'static str {
    WindowsPlatform::ssh_executable()
}

pub fn check_ssh_available() -> Result<bool> {
    WindowsPlatform::check_ssh_available()
}
