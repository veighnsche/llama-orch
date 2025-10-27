//! Daemon binary building
//!
//! TEAM-328: Extracted build logic from install.rs
//! TEAM-329: Moved to dedicated build.rs module (single responsibility)
//!
//! Provides build_daemon() for compiling binaries from source.

use anyhow::Result;
use observability_narration_core::n;

/// Build a daemon binary from source
///
/// TEAM-328: Extracted from install_to_local_bin for clarity
/// TEAM-329: Moved to dedicated build.rs module
///
/// Runs `cargo build --release --bin <binary_name>`
///
/// # Arguments
/// * `binary_name` - Name of the binary to build (e.g., "queen-rbee")
///
/// # Returns
/// * `Ok(String)` - Path to built binary (e.g., "target/release/queen-rbee")
/// * `Err` - Build failed
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::build_daemon;
///
/// # async fn example() -> anyhow::Result<()> {
/// let binary_path = build_daemon("queen-rbee").await?;
/// println!("Built at: {}", binary_path);
/// # Ok(())
/// # }
/// ```
pub async fn build_daemon(binary_name: &str) -> Result<String> {
    n!("build_start", "🔨 Building {} from source...", binary_name);
    
    // Build command
    let mut cmd = std::process::Command::new("cargo");
    cmd.arg("build")
        .arg("--release")
        .arg("--bin")
        .arg(binary_name);
    
    n!("build_exec", "⏳ Running cargo build --release --bin {}...", binary_name);
    let output = cmd.output()?;
    
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        n!("build_failed", "❌ Build failed: {}", stderr);
        anyhow::bail!("Build failed: {}", stderr);
    }
    
    let binary_path = format!("target/release/{}", binary_name);
    n!("build_success", "✅ Build successful: {}", binary_path);
    
    Ok(binary_path)
}
