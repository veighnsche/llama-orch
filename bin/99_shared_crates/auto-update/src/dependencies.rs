// TEAM-309: Extracted dependency parsing logic
//! Dependency graph parsing from Cargo.toml

use anyhow::{Context, Result};
use cargo_toml::Manifest;
use observability_narration_core::n;
use observability_narration_macros::narrate_fn;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// Dependency parser
pub struct DependencyParser;

impl DependencyParser {
    /// Parse all local path dependencies from Cargo.toml (recursive)
    ///
    /// # Arguments
    /// * `workspace_root` - Workspace root directory
    /// * `source_dir` - Source directory relative to workspace root
    ///
    /// # Returns
    /// * `Ok(Vec<PathBuf>)` - List of dependency paths relative to workspace root
    /// * `Err` - Failed to parse dependencies
    #[narrate_fn]
    pub fn parse(workspace_root: &Path, source_dir: &Path) -> Result<Vec<PathBuf>> {
        // TEAM-309: Added narration
        n!("parse_deps", "📦 Parsing dependencies for {}", source_dir.display());

        let mut all_deps = Vec::new();
        let mut visited = HashSet::new();

        Self::collect_recursive(workspace_root, source_dir, &mut all_deps, &mut visited)?;

        n!("parse_deps", "✅ Parsed {} total dependencies (including transitive)", all_deps.len());

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
    #[narrate_fn]
    fn collect_recursive(
        workspace_root: &Path,
        source_dir: &Path,
        all_deps: &mut Vec<PathBuf>,
        visited: &mut HashSet<PathBuf>,
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

        // TEAM-309: Added narration for each Cargo.toml parsed
        n!("parse_cargo_toml", "📄 Parsing {}", cargo_toml.display());

        let manifest = Manifest::from_path(&cargo_toml)
            .with_context(|| format!("Failed to parse {}", cargo_toml.display()))?;

        let mut dep_count = 0;

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
                    dep_count += 1;

                    // Recursively check this dependency's dependencies
                    Self::collect_recursive(workspace_root, &dep_relative, all_deps, visited)?;
                }
            }
        }

        if dep_count > 0 {
            n!(
                "parse_cargo_toml",
                "✅ Found {} local path dependencies in {}",
                dep_count,
                source_dir.display()
            );
        }

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
