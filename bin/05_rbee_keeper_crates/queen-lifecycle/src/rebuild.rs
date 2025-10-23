//! Queen rebuild operation
//!
//! TEAM-276: Extracted from rbee-keeper/src/handlers/queen.rs
//! TEAM-262: Added queen rebuild command for local-hive optimization
//! TEAM-263: Implemented actual build logic

use anyhow::Result;
use observability_narration_core::NarrationFactory;

const NARRATE: NarrationFactory = NarrationFactory::new("queen-life");

/// Rebuild queen-rbee with optional features
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
    NARRATE.action("queen_rebuild").human("🔨 Rebuilding queen-rbee...").emit();

    // Determine build command
    let mut cmd = std::process::Command::new("cargo");
    cmd.arg("build").arg("--release").arg("--bin").arg("queen-rbee");

    if with_local_hive {
        NARRATE
            .action("queen_rebuild")
            .human("✨ Building with integrated local hive (50-100x faster localhost)...")
            .emit();
        cmd.arg("--features").arg("local-hive");
    } else {
        NARRATE
            .action("queen_rebuild")
            .human("📡 Building distributed queen (remote hives only)...")
            .emit();
    }

    // Execute build
    NARRATE
        .action("queen_rebuild")
        .human("⏳ Running cargo build (this may take a few minutes)...")
        .emit();

    let output = cmd.output()?;

    if output.status.success() {
        NARRATE.action("queen_rebuild").human("✅ Build successful!").emit();

        // Show binary location
        let binary_path = "target/release/queen-rbee";
        NARRATE
            .action("queen_rebuild")
            .context(binary_path)
            .human("📦 Binary available at: {}")
            .emit();

        if with_local_hive {
            NARRATE
                .action("queen_rebuild")
                .human("💡 Restart queen to use the new binary with local-hive feature")
                .emit();
        }
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        NARRATE
            .action("queen_rebuild")
            .context(stderr.to_string())
            .human("❌ Build failed: {}")
            .error_kind("build_failed")
            .emit();
        anyhow::bail!("Build failed");
    }
}
