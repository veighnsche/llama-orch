// TEAM-309: Extracted rebuild checking logic
//! Rebuild necessity checker

use anyhow::{Context, Result};
use observability_narration_core::n;
use observability_narration_macros::narrate_fn;

use std::path::Path;
use std::time::SystemTime;
use walkdir::WalkDir;

use crate::updater::AutoUpdater;

/// Rebuild checker
pub struct RebuildChecker;

impl RebuildChecker {
    /// Check if binary needs rebuilding
    ///
    /// # Returns
    /// * `Ok(true)` - Rebuild needed
    /// * `Ok(false)` - Binary is up-to-date
    /// * `Err` - Failed to check
    #[narrate_fn]
    pub fn check(updater: &AutoUpdater) -> Result<bool> {
        // TEAM-309: Added narration
        n!("check_rebuild", "🔍 Checking if {} needs rebuild", updater.binary_name);

        // Find binary (debug or release)
        let binary_path = match crate::binary::BinaryFinder::find_path(updater) {
            Ok(path) => path,
            Err(_) => {
                n!("check_rebuild", "⚠️  Binary {} not found, rebuild needed", updater.binary_name);
                return Ok(true);
            }
        };

        let binary_time =
            std::fs::metadata(&binary_path)?.modified().context("Failed to get binary mtime")?;

        // TEAM-309: Added narration for binary timestamp
        n!("check_rebuild", "📅 Binary {} last modified: {:?}", updater.binary_name, binary_time);

        // Check binary's own source directory
        let source_path = updater.workspace_root.join(&updater.source_dir);
        if Self::is_dir_newer(&source_path, binary_time)? {
            n!(
                "check_rebuild",
                "🔨 Source directory {} changed, rebuild needed",
                updater.source_dir.display()
            );
            return Ok(true);
        }

        // Check all dependencies
        for dep_path in &updater.dependencies {
            let full_path = updater.workspace_root.join(dep_path);
            if Self::is_dir_newer(&full_path, binary_time)? {
                n!("check_rebuild", "🔨 Dependency {} changed, rebuild needed", dep_path.display());
                return Ok(true);
            }
        }

        n!("check_rebuild", "✅ Binary {} is up-to-date", updater.binary_name);

        Ok(false)
    }

    /// Check if any file in directory is newer than reference time
    ///
    /// # Arguments
    /// * `dir` - Directory to scan
    /// * `reference_time` - Reference timestamp
    ///
    /// # Returns
    /// * `Ok(true)` - Directory has newer files
    /// * `Ok(false)` - No newer files found
    /// * `Err` - Failed to scan directory
    #[narrate_fn]
    fn is_dir_newer(dir: &Path, reference_time: SystemTime) -> Result<bool> {
        if !dir.exists() {
            return Ok(false);
        }

        let mut files_checked = 0;
        let mut newer_found = false;

        for entry in WalkDir::new(dir).follow_links(false) {
            let entry = entry?;
            let path = entry.path();

            // Skip target directories
            if path.components().any(|c| c.as_os_str() == "target") {
                continue;
            }

            // Check .rs files and Cargo.toml
            if let Some(ext) = path.extension() {
                if ext == "rs" || path.file_name().map(|n| n == "Cargo.toml").unwrap_or(false) {
                    files_checked += 1;
                    let meta = std::fs::metadata(path)?;
                    if let Ok(modified) = meta.modified() {
                        if modified > reference_time {
                            // TEAM-309: Added narration for first newer file found
                            n!("file_changed", "📝 File {} is newer than binary", path.display());
                            newer_found = true;
                            break;
                        }
                    }
                }
            }
        }

        // TEAM-309: Added narration for scan results
        if !newer_found && files_checked > 0 {
            n!(
                "scan_complete",
                "✅ Scanned {} files in {}, none newer",
                files_checked,
                dir.display()
            );
        }

        Ok(newer_found)
    }
}
