//! Queen rebuild operation
//!
//! TEAM-276: Extracted from rbee-keeper/src/handlers/queen.rs
//! TEAM-262: Added queen rebuild command for local-hive optimization
//! TEAM-263: Implemented actual build logic
//! TEAM-312: Migrated to n!() macro, added running check

use anyhow::Result;
use observability_narration_core::n;

/// Rebuild queen-rbee with optional features (update operation)
///
/// TEAM-296: This is the 'update' command - rebuilds from source
/// TEAM-312: Added check to prevent rebuilding while queen is running
///
/// Runs `cargo build --release --bin queen-rbee` with optional features.
///
/// # Arguments
/// * `with_local_hive` - Include local-hive feature for 50-100x faster localhost operations
///
/// # Returns
/// * `Ok(())` - Build successful
/// * `Err` - Build failed
pub async fn rebuild_queen(with_local_hive: bool) -> Result<()> {
    n!("start", "🔄 Updating queen-rbee (rebuilding from source)...");
    
    // TEAM-312: Check if queen is running (same as uninstall)
    let queen_url = "http://localhost:7833";
    let is_running = daemon_lifecycle::health::is_daemon_healthy(
        queen_url,
        None, // Use default /health endpoint
        Some(std::time::Duration::from_secs(2)),
    )
    .await;
    
    if is_running {
        n!("daemon_still_running", "⚠️  Queen is currently running. Stop it first.");
        anyhow::bail!("Queen is still running. Use 'rbee queen stop' first.");
    }

    // Determine build command
    let mut cmd = std::process::Command::new("cargo");
    cmd.arg("build").arg("--release").arg("--bin").arg("queen-rbee");

    if with_local_hive {
        n!("build_mode", "✨ Building with integrated local hive (50-100x faster localhost)...");
        cmd.arg("--features").arg("local-hive");
    } else {
        n!("build_mode", "📡 Building distributed queen (remote hives only)...");
    }

    // Execute build
    n!("build_start", "⏳ Running cargo build (this may take a few minutes)...");

    let output = cmd.output()?;

    if output.status.success() {
        n!("build_success", "✅ Build successful!");

        // Show binary location
        let binary_path = "target/release/queen-rbee";
        n!("binary_location", "📦 Binary available at: {}", binary_path);

        if with_local_hive {
            n!("restart_hint", "💡 Restart queen to use the new binary with local-hive feature");
        }
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        n!("build_failed", "❌ Build failed: {}", stderr);
        anyhow::bail!("Build failed");
    }
}
