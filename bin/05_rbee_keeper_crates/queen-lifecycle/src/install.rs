//! Queen install operation
//!
//! TEAM-276: Extracted from rbee-keeper/src/handlers/queen.rs
//! TEAM-262: Install queen binary
//! TEAM-263: Implemented install logic

use anyhow::Result;
use observability_narration_core::NarrationFactory;

const NARRATE: NarrationFactory = NarrationFactory::new("queen-life");

/// Install queen-rbee binary to ~/.local/bin
///
/// TEAM-296: Enhanced to check if already installed
///
/// Steps:
/// 1. Check if already installed (error if yes)
/// 2. Build queen-rbee from git repo (cargo build --release)
/// 3. Copy to ~/.local/bin/queen-rbee
/// 4. Make executable (Unix only)
///
/// # Arguments
/// * `binary` - Optional custom binary path (if None, builds from source)
///
/// # Returns
/// * `Ok(())` - Installation successful
/// * `Err` - Already installed, build failed, or installation failed
pub async fn install_queen(binary: Option<String>) -> Result<()> {
    NARRATE.action("queen_install").human("📦 Installing queen-rbee...").emit();

    // TEAM-296: Check if already installed
    let home = std::env::var("HOME")?;
    let install_path = std::path::PathBuf::from(format!("{}/.local/bin/queen-rbee", home));
    
    if install_path.exists() {
        NARRATE
            .action("queen_install")
            .context(install_path.display().to_string())
            .human("❌ Queen already installed at: {}")
            .error_kind("already_installed")
            .emit();
        anyhow::bail!("Queen already installed. Use 'queen update' to rebuild or 'queen uninstall' first.");
    }

    // TEAM-296: Build from source if no binary path provided
    let source_path = if let Some(binary_path) = binary {
        // Use provided binary path
        let path = std::path::PathBuf::from(&binary_path);
        if !path.exists() {
            anyhow::bail!("Binary not found at: {}", binary_path);
        }
        path
    } else {
        // Build from source
        NARRATE
            .action("queen_install")
            .human("🔨 Building queen-rbee from source (cargo build --release)...")
            .emit();
        
        let output = std::process::Command::new("cargo")
            .args(["build", "--release", "--bin", "queen-rbee"])
            .output()?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            NARRATE
                .action("queen_install")
                .context(stderr.to_string())
                .human("❌ Build failed: {}")
                .error_kind("build_failed")
                .emit();
            anyhow::bail!("Build failed");
        }
        
        NARRATE.action("queen_install").human("✅ Build successful!").emit();
        std::path::PathBuf::from("target/release/queen-rbee")
    };

    // Determine install location (~/.local/bin/queen-rbee)
    let install_dir = std::path::PathBuf::from(format!("{}/.local/bin", home));

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
