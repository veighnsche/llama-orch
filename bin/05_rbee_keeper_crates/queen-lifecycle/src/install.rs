//! Queen install operation
//!
//! TEAM-276: Extracted from rbee-keeper/src/handlers/queen.rs
//! TEAM-262: Install queen binary
//! TEAM-263: Implemented install logic

use anyhow::Result;
use daemon_lifecycle::{install_daemon, InstallConfig};
use observability_narration_core::NarrationFactory;

const NARRATE: NarrationFactory = NarrationFactory::new("queen-life");

/// Install queen-rbee binary to ~/.local/bin
///
/// Steps:
/// 1. Use daemon-lifecycle to find/validate binary (target/release or target/debug)
/// 2. Copy to ~/.local/bin/queen-rbee
/// 3. Make executable (Unix only)
///
/// # Arguments
/// * `binary` - Optional custom binary path (if None, auto-detects from target/)
///
/// # Returns
/// * `Ok(())` - Installation successful
/// * `Err` - Binary not found or installation failed
pub async fn install_queen(binary: Option<String>) -> Result<()> {
    NARRATE.action("queen_install").human("📦 Installing queen-rbee...").emit();

    // Use daemon-lifecycle to find/validate binary
    let config = InstallConfig {
        binary_name: "queen-rbee".to_string(),
        binary_path: binary,
        target_path: None,
        job_id: None,
    };

    let install_result = install_daemon(config).await?;
    let source_path = std::path::PathBuf::from(&install_result.binary_path);

    // Determine install location (~/.local/bin/queen-rbee)
    let home = std::env::var("HOME")?;
    let install_dir = std::path::PathBuf::from(format!("{}/.local/bin", home));
    let install_path = install_dir.join("queen-rbee");

    // Create install directory if needed
    std::fs::create_dir_all(&install_dir)?;

    // Copy binary
    NARRATE
        .action("queen_install")
        .context(install_path.display().to_string())
        .human("📋 Installing to: {}")
        .emit();

    std::fs::copy(&source_path, &install_path)?;

    // Make executable (Unix only)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&install_path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&install_path, perms)?;
    }

    NARRATE.action("queen_install").human("✅ Queen installed successfully!").emit();
    NARRATE
        .action("queen_install")
        .context(install_path.display().to_string())
        .human("📍 Binary location: {}")
        .emit();
    NARRATE.action("queen_install").human("💡 Make sure ~/.local/bin is in your PATH").emit();

    Ok(())
}
