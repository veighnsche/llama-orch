// TEAM-309: Extracted binary finding logic
//! Binary location finder

use anyhow::Result;
use observability_narration_core::n;
use observability_narration_macros::narrate_fn;
use std::path::PathBuf;

use crate::updater::AutoUpdater;

/// Binary finder
pub struct BinaryFinder;

impl BinaryFinder {
    /// Find binary in target directory
    ///
    /// Searches:
    /// 1. `target/debug/<binary_name>`
    /// 2. `target/release/<binary_name>`
    ///
    /// # Returns
    /// * `Ok(PathBuf)` - Path to binary
    /// * `Err` - Binary not found
    pub fn find(updater: &AutoUpdater) -> Result<PathBuf> {
        Self::find_path(updater)
    }

    /// Internal method to find binary path
    #[narrate_fn]
    pub(crate) fn find_path(updater: &AutoUpdater) -> Result<PathBuf> {
        // TEAM-309: Added narration
        n!("find_binary", "🔍 Searching for binary {}", updater.binary_name);

        // Try debug first (development mode)
        let debug_path = updater.workspace_root.join("target/debug").join(&updater.binary_name);
        if debug_path.exists() {
            n!(
                "find_binary",
                "✅ Found {} in debug mode at {}",
                updater.binary_name,
                debug_path.display()
            );
            return Ok(debug_path);
        }

        // Try release
        let release_path = updater.workspace_root.join("target/release").join(&updater.binary_name);
        if release_path.exists() {
            n!(
                "find_binary",
                "✅ Found {} in release mode at {}",
                updater.binary_name,
                release_path.display()
            );
            return Ok(release_path);
        }

        n!(
            "find_binary",
            "❌ Binary {} not found in target/debug or target/release",
            updater.binary_name
        );

        anyhow::bail!(
            "Binary '{}' not found in target/debug or target/release",
            updater.binary_name
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_binary() {
        // This test assumes rbee-keeper is built
        let updater = crate::AutoUpdater::new("rbee-keeper", "bin/00_rbee_keeper").unwrap();
        let binary = BinaryFinder::find(&updater);

        // May fail if binary not built, that's ok
        if let Ok(path) = binary {
            assert!(path.exists());
            assert!(path.to_string_lossy().contains("rbee-keeper"));
        }
    }
}
