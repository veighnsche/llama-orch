// TEAM-309: Extracted rebuild logic
//! Binary rebuilding via cargo

use anyhow::{anyhow, Context, Result};
use observability_narration_core::n;
use observability_narration_macros::narrate_fn;
use std::process::Command;

use crate::updater::AutoUpdater;

/// Binary rebuilder
pub struct Rebuilder;

impl Rebuilder {
    /// Rebuild the binary
    ///
    /// Runs `cargo build --bin <binary_name>` in workspace root.
    ///
    /// # Returns
    /// * `Ok(())` - Build succeeded
    /// * `Err` - Build failed
    #[narrate_fn]
    pub fn rebuild(updater: &AutoUpdater) -> Result<()> {
        // TEAM-309: Added narration
        n!("start", 
            human: "🔨 Rebuilding {}...",
            cute: "🐝 Building {} with love!",
            story: "The keeper commanded: 'Build {}'",
            updater.binary_name
        );

        let start_time = std::time::Instant::now();

        let status = Command::new("cargo")
            .arg("build")
            .arg("--bin")
            .arg(&updater.binary_name)
            .current_dir(&updater.workspace_root)
            .status()
            .context("Failed to run cargo build")?;

        if !status.success() {
            n!("failed", "❌ Failed to rebuild {}", updater.binary_name);
            return Err(anyhow!("Failed to rebuild {}", updater.binary_name));
        }

        let elapsed_ms = start_time.elapsed().as_millis() as u64;
        let elapsed_secs = elapsed_ms as f64 / 1000.0;

        // TEAM-309: Enhanced narration with timing and all 3 modes
        n!("success",
            human: "✅ Rebuilt {} successfully in {:.2}s",
            cute: "🎉 {} is ready! Built in {:.2}s!",
            story: "'Your binary {} is ready', whispered the compiler after {:.2}s",
            updater.binary_name,
            elapsed_secs
        );

        Ok(())
    }
}
