// TEAM-309: Extracted dependency parsing logic
//! Dependency graph parsing from Cargo.toml

use anyhow::{Context, Result};
use cargo_toml::Manifest;
use observability_narration_core::n;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Dependency parser
pub struct DependencyParser;

impl DependencyParser {
    /// Parse all local path dependencies from Cargo.toml (recursive)
    ///
    /// TEAM-311: Phase 3 - Dependencies with batching
    ///
    /// # Arguments
    /// * `workspace_root` - Workspace root directory
    /// * `source_dir` - Source directory relative to workspace root
    ///
    /// # Returns
    /// * `Ok(Vec<PathBuf>)` - List of dependency paths relative to workspace root
    /// * `Err` - Failed to parse dependencies
    pub fn parse(workspace_root: &Path, source_dir: &Path) -> Result<Vec<PathBuf>> {
        // TEAM-311: Phase 3 - Dependencies
        let start = Instant::now();
        n!("phase_deps", "📦 Dependency discovery");
        n!("parse_deps", "Scanning root crate: {}", source_dir.display());

        let mut all_deps = Vec::new();
        let mut visited = HashSet::new();
        let mut toml_files = Vec::new();

        Self::collect_recursive(
            workspace_root,
            source_dir,
            &mut all_deps,
            &mut visited,
            &mut toml_files,
        )?;

        // TEAM-311: Batch summary
        n!("collect_tomls", "Queued Cargo.toml files: {}", toml_files.len());

        let local_deps = all_deps.len();
        let transitive_deps = toml_files.len().saturating_sub(1); // Root crate doesn't count as transitive
        n!(
            "parse_batch",
            "Parsed {} deps · {} local path · {} transitive",
            toml_files.len(),
            local_deps,
            transitive_deps
        );

        // TEAM-312: Removed nd!() debug logging (entropy)

        let elapsed = start.elapsed().as_millis();
        n!("summary", "✅ Deps ok · {}ms", elapsed);

        Ok(all_deps)
    }

    // ============================================================
    // BUG FIX: TEAM-260 | Dependency paths not resolved correctly
    // ============================================================
    // SUSPICION:
    // - AutoUpdater was not detecting dependency changes
    // - Changed narration-core/src/lib.rs but rebuild not triggered
    // - File timestamps showed source newer than binary
    //
    // INVESTIGATION:
    // - Checked how dependency paths are parsed from Cargo.toml
    // - Found paths are relative to crate directory (e.g., "../99_shared_crates/narration-core")
    // - These relative paths were stored as-is in all_deps
    // - In needs_rebuild(), code does: workspace_root.join(dep_path)
    // - This creates WRONG paths like: workspace_root/../99_shared_crates/narration-core
    // - The "../" puts it OUTSIDE the workspace, so is_dir_newer never finds the files
    //
    // ROOT CAUSE:
    // - Dependency paths from Cargo.toml are relative to the CRATE directory
    // - But we were joining them directly to workspace_root
    // - This created invalid paths that pointed outside the workspace
    //
    // FIX:
    // - Resolve dependency path relative to crate's directory first
    // - Convert absolute path back to relative from workspace_root
    // - Store the workspace-relative path in all_deps
    // - Example: bin/00_rbee_keeper + ../99_shared_crates/narration-core
    //            → workspace_root/bin/00_rbee_keeper/../99_shared_crates/narration-core
    //            → canonicalize → workspace_root/bin/99_shared_crates/narration-core
    //            → strip workspace_root → bin/99_shared_crates/narration-core
    //
    // TESTING:
    // - Modified narration-core/src/lib.rs (added comment)
    // - Ran ./rbee hive list
    // - Verified auto-update detects change and triggers rebuild
    // - Tested with multiple dependency levels (transitive dependencies)
    // - All dependency changes now properly detected
    // ============================================================

    /// Recursively collect dependencies from Cargo.toml
    ///
    /// TEAM-311: Modified to track toml files for batching
    fn collect_recursive(
        workspace_root: &Path,
        source_dir: &Path,
        all_deps: &mut Vec<PathBuf>,
        visited: &mut HashSet<PathBuf>,
        toml_files: &mut Vec<(PathBuf, usize, usize)>, // (path, local_count, transitive_count)
    ) -> Result<()> {
        let normalized = source_dir.to_path_buf();
        if visited.contains(&normalized) {
            return Ok(());
        }
        visited.insert(normalized.clone());

        let cargo_toml = workspace_root.join(source_dir).join("Cargo.toml");
        if !cargo_toml.exists() {
            return Ok(());
        }

        let manifest = Manifest::from_path(&cargo_toml)
            .with_context(|| format!("Failed to parse {}", cargo_toml.display()))?;

        let mut local_count = 0;
        let mut transitive_count = 0;

        // Parse [dependencies]
        for (_name, dep) in manifest.dependencies {
            if let Some(detail) = dep.detail() {
                if let Some(path) = &detail.path {
                    // TEAM-260: FIX - Resolve dependency path correctly
                    // Path from Cargo.toml is relative to the crate's directory
                    // Example: if source_dir = "bin/00_rbee_keeper"
                    //          and path = "../99_shared_crates/narration-core"
                    // We need to resolve: workspace_root/bin/00_rbee_keeper/../99_shared_crates/narration-core
                    // Then make it relative to workspace_root again

                    let crate_dir = workspace_root.join(source_dir);
                    let dep_absolute = crate_dir.join(path);

                    // Canonicalize to resolve ".." components
                    let dep_canonical = dep_absolute.canonicalize().with_context(|| {
                        format!("Failed to resolve dependency path: {}", dep_absolute.display())
                    })?;

                    // Convert back to relative path from workspace_root
                    let dep_relative = dep_canonical
                        .strip_prefix(workspace_root)
                        .with_context(|| {
                            format!("Dependency {} is outside workspace", dep_canonical.display())
                        })?
                        .to_path_buf();

                    all_deps.push(dep_relative.clone());
                    local_count += 1;

                    // Recursively check this dependency's dependencies
                    let before_len = all_deps.len();
                    Self::collect_recursive(
                        workspace_root,
                        &dep_relative,
                        all_deps,
                        visited,
                        toml_files,
                    )?;
                    transitive_count += all_deps.len() - before_len;
                }
            }
        }

        // TEAM-311: Track this toml file for debug output
        toml_files.push((cargo_toml, local_count, transitive_count));

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workspace::WorkspaceFinder;

    #[test]
    fn test_parse_dependencies() {
        let root = WorkspaceFinder::find().unwrap();
        let deps = DependencyParser::parse(&root, &PathBuf::from("bin/00_rbee_keeper")).unwrap();

        // rbee-keeper should have dependencies
        assert!(!deps.is_empty());

        // Should include daemon-lifecycle
        assert!(deps.iter().any(|d| d.to_string_lossy().contains("daemon-lifecycle")));

        // Should include narration-core
        assert!(deps.iter().any(|d| d.to_string_lossy().contains("narration-core")));
    }
}
