// TEAM-309: Extracted rebuild checking logic
//! Rebuild necessity checker

use anyhow::{Context, Result};
use observability_narration_core::n;
use observability_narration_macros::narrate_fn;

use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime};
use walkdir::WalkDir;

use crate::updater::AutoUpdater;

/// Rebuild checker
pub struct RebuildChecker;

impl RebuildChecker {
    /// Check if binary needs rebuilding
    ///
    /// TEAM-311: Phases 4-6 (Build State, File Scans, Decision)
    ///
    /// # Returns
    /// * `Ok(true)` - Rebuild needed
    /// * `Ok(false)` - Binary is up-to-date
    /// * `Err` - Failed to check
    #[narrate_fn]
    pub fn check(updater: &AutoUpdater) -> Result<bool> {
        // TEAM-311: Phase 4 - Build State
        let start_phase4 = Instant::now();
        n!("phase_build", "🛠️ Build state");

        // Find binary (debug or release)
        let (binary_path, build_mode) = match crate::binary::BinaryFinder::find_path(updater) {
            Ok(path) => {
                let mode =
                    if path.to_string_lossy().contains("debug") { "debug" } else { "release" };
                (path, mode)
            }
            Err(_) => {
                n!("check_rebuild", "Binary: not found");
                n!("find_binary", "Mode=unknown · found=none");

                let elapsed = start_phase4.elapsed().as_millis();
                n!("summary", "✅ Build state ok · {}ms", elapsed);

                // TEAM-311: Phase 6 - Decision (binary not found)
                let start_phase6 = Instant::now();
                n!("phase_decision", "📑 Rebuild decision");
                n!("needs_rebuild", "⚠️ Rebuild required · Binary not found");

                let elapsed = start_phase6.elapsed().as_millis();
                n!("summary", "⚠️ Rebuild needed · {}ms", elapsed);

                return Ok(true);
            }
        };

        let binary_time =
            std::fs::metadata(&binary_path)?.modified().context("Failed to get binary mtime")?;

        let mtime_display = format_timestamp(binary_time);
        n!("check_rebuild", "Binary: {} · mtime: {}", binary_path.display(), mtime_display);
        n!("find_binary", "Mode={} · found={}", build_mode, updater.binary_name);

        let elapsed = start_phase4.elapsed().as_millis();
        n!("summary", "✅ Build state ok · {}ms", elapsed);

        // TEAM-311: Phase 5 - File Scans
        let start_phase5 = Instant::now();
        n!("phase_scan", "🔍 Source freshness checks");

        // Collect all directories to scan (deduplicated)
        let mut dirs_to_scan = Vec::new();
        dirs_to_scan.push(updater.workspace_root.join(&updater.source_dir));
        for dep_path in &updater.dependencies {
            dirs_to_scan.push(updater.workspace_root.join(dep_path));
        }

        // Scan and emit one line per directory
        let mut total_files = 0;
        let mut total_newer = 0;
        let mut first_newer_dir: Option<(PathBuf, usize)> = None;

        for dir in &dirs_to_scan {
            let (files, newer) = Self::scan_directory(dir, binary_time)?;

            let rel_dir = dir.strip_prefix(&updater.workspace_root).unwrap_or(dir);
            n!("is_dir_newer", "{} · files={} · newer={}", rel_dir.display(), files, newer);

            total_files += files;
            total_newer += newer;

            if newer > 0 && first_newer_dir.is_none() {
                first_newer_dir = Some((rel_dir.to_path_buf(), newer));
            }
        }

        let elapsed = start_phase5.elapsed().as_millis();
        n!(
            "summary",
            "Scanned {} dirs · {} files · newer={} · {}ms",
            dirs_to_scan.len(),
            total_files,
            total_newer,
            elapsed
        );

        // TEAM-311: Phase 6 - Decision
        let start_phase6 = Instant::now();
        n!("phase_decision", "📑 Rebuild decision");

        let needs_rebuild = total_newer > 0;

        if needs_rebuild {
            let reason = if let Some((dir, count)) = first_newer_dir {
                format!("{} has {} newer files", dir.display(), count)
            } else {
                format!("{} directories have newer files", dirs_to_scan.len())
            };
            n!("needs_rebuild", "⚠️ Rebuild required · {}", reason);
        } else {
            n!("up_to_date", "✅ {} is up-to-date", updater.binary_name);
        }

        let elapsed = start_phase6.elapsed().as_millis();
        let status = if needs_rebuild { "⚠️" } else { "✅ No" };
        n!("summary", "{} rebuild needed · {}ms", status, elapsed);

        Ok(needs_rebuild)
    }

    /// Scan directory and return file counts
    ///
    /// TEAM-311: Modified to return counts instead of boolean
    ///
    /// # Arguments
    /// * `dir` - Directory to scan
    /// * `reference_time` - Reference timestamp
    ///
    /// # Returns
    /// * `Ok((total_files, newer_files))` - File counts
    /// * `Err` - Failed to scan directory
    fn scan_directory(dir: &Path, reference_time: SystemTime) -> Result<(usize, usize)> {
        if !dir.exists() {
            return Ok((0, 0));
        }

        let mut total_files = 0;
        let mut newer_files = 0;

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
                    total_files += 1;
                    let meta = std::fs::metadata(path)?;
                    if let Ok(modified) = meta.modified() {
                        if modified > reference_time {
                            newer_files += 1;
                        }
                    }
                }
            }
        }

        Ok((total_files, newer_files))
    }
}

/// Format timestamp for display
///
/// TEAM-311: Formats as ISO 8601 or seconds since epoch
fn format_timestamp(time: SystemTime) -> String {
    match time.duration_since(SystemTime::UNIX_EPOCH) {
        Ok(duration) => {
            let secs = duration.as_secs();
            // Simple ISO-like format: YYYY-MM-DD HH:MM:SS
            format!("{} seconds since epoch", secs)
        }
        Err(_) => "unknown".to_string(),
    }
}
