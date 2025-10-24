//! Install rbee-hive on local or remote host
//!
//! TEAM-290: Local or remote hive installation

use anyhow::{Context, Result};
use daemon_lifecycle::{install_daemon, InstallConfig};
use observability_narration_core::NarrationFactory;
use std::path::Path;

use crate::ssh::SshClient;

const NARRATE: NarrationFactory = NarrationFactory::new("hive-inst");

/// Install rbee-hive on local or remote host
///
/// TEAM-290: Supports both local and remote installation
///
/// # Arguments
/// * `host` - Host to install on ("localhost" for local, SSH alias for remote)
/// * `binary_path` - Optional path to rbee-hive binary (auto-detects if None)
/// * `install_dir` - Installation directory (default: ~/.local/bin for localhost, /usr/local/bin for remote)
///
/// # Example
/// ```rust,ignore
/// // Install locally
/// install_hive("localhost", None, None).await?;
///
/// // Install remotely
/// install_hive("gpu-server", Some("./rbee-hive"), Some("/usr/local/bin")).await?;
/// ```
pub async fn install_hive(
    host: &str,
    binary_path: Option<String>,
    install_dir: Option<String>,
) -> Result<()> {
    NARRATE
        .action("install_hive_start")
        .context(host)
        .human("📦 Installing rbee-hive on '{}'")
        .emit();

    // Check if localhost (direct install) or remote (SSH install)
    if host == "localhost" || host == "127.0.0.1" {
        install_hive_local(binary_path, install_dir).await
    } else {
        install_hive_remote(host, binary_path, install_dir).await
    }
}

/// Install rbee-hive locally (no SSH)
async fn install_hive_local(
    binary_path: Option<String>,
    install_dir: Option<String>,
) -> Result<()> {
    NARRATE
        .action("install_hive_local")
        .human("📦 Installing rbee-hive locally...")
        .emit();

    // Use daemon-lifecycle to find/validate binary
    let config = InstallConfig {
        binary_name: "rbee-hive".to_string(),
        binary_path,
        target_path: None,
        job_id: None,
    };

    let install_result = install_daemon(config).await?;
    let source_path = std::path::PathBuf::from(&install_result.binary_path);

    // Determine install location
    let install_dir = if let Some(dir) = install_dir {
        std::path::PathBuf::from(dir)
    } else {
        // Default: ~/.local/bin
        let home = std::env::var("HOME")?;
        std::path::PathBuf::from(format!("{}/.local/bin", home))
    };
    let install_path = install_dir.join("rbee-hive");

    // Create install directory if needed
    std::fs::create_dir_all(&install_dir)?;

    // Copy binary
    std::fs::copy(&source_path, &install_path)
        .context("Failed to copy hive binary")?;

    // Make executable (Unix only)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&install_path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&install_path, perms)?;
    }

    // Verify installation
    let output = std::process::Command::new(&install_path)
        .arg("--version")
        .output()
        .context("Failed to verify hive installation")?;

    let version = String::from_utf8_lossy(&output.stdout);

    NARRATE
        .action("install_hive_complete")
        .context(install_path.display().to_string())
        .context(version.trim())
        .human("✅ Hive installed at '{}': {}")
        .emit();

    Ok(())
}

/// Install rbee-hive remotely via SSH
async fn install_hive_remote(
    host: &str,
    binary_path: Option<String>,
    install_dir: Option<String>,
) -> Result<()> {
    NARRATE
        .action("install_hive_remote")
        .context(host)
        .human("📦 Installing rbee-hive on '{}' via SSH...")
        .emit();

    // Binary path is required for remote install
    let binary_path = binary_path
        .ok_or_else(|| anyhow::anyhow!("Binary path required for remote install"))?;

    // Verify local binary exists
    if !Path::new(&binary_path).exists() {
        anyhow::bail!("Hive binary not found: {}", binary_path);
    }

    // Connect via SSH
    let client = SshClient::connect(host).await?;

    // Default remote install dir
    let install_dir = install_dir.unwrap_or_else(|| "/usr/local/bin".to_string());

    // Create install directory
    client
        .execute(&format!("mkdir -p {}", install_dir))
        .await
        .context("Failed to create install directory")?;

    // Upload binary
    let remote_path = format!("{}/rbee-hive", install_dir);
    client
        .upload_file(&binary_path, &remote_path)
        .await
        .context("Failed to upload hive binary")?;

    // Make executable
    client
        .execute(&format!("chmod +x {}", remote_path))
        .await
        .context("Failed to make hive executable")?;

    // Verify installation
    let version = client
        .execute(&format!("{} --version", remote_path))
        .await
        .context("Failed to verify hive installation")?;

    NARRATE
        .action("install_hive_complete")
        .context(host)
        .context(version.trim())
        .human("✅ Hive installed on '{}': {}")
        .emit();

    Ok(())
}
