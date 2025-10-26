//! Queen install operation
//!
//! TEAM-276: Extracted from rbee-keeper/src/handlers/queen.rs
//! TEAM-262: Install queen binary
//! TEAM-263: Implemented install logic
//! TEAM-311: Migrated to n!() macro

use anyhow::Result;
use observability_narration_core::n;

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
/// # Errors
/// This function will return an error if:
/// * HOME environment variable is not set
/// * Queen is already installed at ~/.local/bin/queen-rbee
/// * Provided binary path does not exist
/// * Cargo build fails (when building from source)
/// * Failed to create installation directory
/// * Failed to copy binary to installation path
/// * Failed to set executable permissions (Unix only)
pub async fn install_queen(binary: Option<String>) -> Result<()> {
    n!("start", "📦 Installing queen-rbee...");

    // TEAM-296: Check if already installed
    let home = std::env::var("HOME")?;
    let install_path = std::path::PathBuf::from(format!("{}/.local/bin/queen-rbee", home));

    if install_path.exists() {
        n!("exists", "❌ Queen already installed at: {}", install_path.display());
        anyhow::bail!(
            "Queen already installed. Use 'queen update' to rebuild or 'queen uninstall' first."
        );
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
        n!("building", "🔨 Building queen-rbee from source (cargo build --release)...");

        // TEAM-312: Use .status() instead of .output() to stream cargo output to terminal
        let status = std::process::Command::new("cargo")
            .args(["build", "--release", "--bin", "queen-rbee"])
            .status()?;

        if !status.success() {
            n!("failed", "❌ Build failed");
            anyhow::bail!("Build failed");
        }

        n!("success", "✅ Build successful!");
        std::path::PathBuf::from("target/release/queen-rbee")
    };

    // Determine install location (~/.local/bin/queen-rbee)
    let install_dir = std::path::PathBuf::from(format!("{}/.local/bin", home));

    // Create install directory if needed
    std::fs::create_dir_all(&install_dir)?;

    // Copy binary
    n!("move", "📋 Installing to: {}", install_path.display());

    std::fs::copy(&source_path, &install_path)?;

    // Make executable (Unix only)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&install_path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&install_path, perms)?;
    }

    n!("done", "✅ Queen installed successfully!");
    n!("info", "📍 Binary location: {}", install_path.display());
    n!("info", "💡 Make sure ~/.local/bin is in your PATH");

    Ok(())
}
