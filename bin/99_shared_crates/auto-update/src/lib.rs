// TEAM-193: Created by TEAM-193
// Purpose: Shared auto-update logic with full dependency tracking
//
// CRITICAL: This crate is closely coupled with:
// - daemon-lifecycle (bin/99_shared_crates/daemon-lifecycle/)
// - hive-lifecycle (bin/15_queen_rbee_crates/hive-lifecycle/)
// - worker-lifecycle (bin/25_rbee_hive_crates/worker-lifecycle/)
//
// These lifecycle crates spawn daemons and need auto-update to ensure
// binaries are rebuilt when dependencies change.

#![warn(missing_docs)]
#![warn(clippy::all)]

//! auto-update
//!
//! **Category:** Utility
//! **Pattern:** Builder Pattern
//! **Standard:** See `/bin/CRATE_INTERFACE_STANDARD.md`
//!
//! Dependency-aware auto-update logic for rbee binaries.
//!
//! # Problem
//!
//! When a shared crate is edited (e.g., `daemon-lifecycle`), all binaries
//! that depend on it need to be rebuilt. Simple mtime checks on the binary's
//! source directory miss these transitive dependencies.
//!
//! # Solution
//!
//! Parse `Cargo.toml` to find ALL local path dependencies, check them recursively,
//! and trigger rebuild if ANY dependency changed.
//!
//! # Interface
//!
//! ## Builder Pattern
//! ```rust,ignore
//! use auto_update::AutoUpdater;
//!
//! let updater = AutoUpdater::new("queen-rbee", "bin/10_queen_rbee")?;
//!
//! // Check if rebuild needed
//! if updater.needs_rebuild()? {
//!     updater.rebuild()?;
//! }
//!
//! // Or: one-shot ensure built
//! let binary_path = updater.ensure_built().await?;
//! ```
//!
//! # Lifecycle Integration
//!
//! This crate is designed to be used by lifecycle crates:
//!
//! ```rust,ignore
//! // In daemon-lifecycle (keeper → queen)
//! use auto_update::AutoUpdater;
//!
//! pub async fn spawn_queen(config: &Config) -> Result<Child> {
//!     if config.auto_update_queen {
//!         AutoUpdater::new("queen-rbee", "bin/10_queen_rbee")?
//!             .ensure_built()
//!             .await?;
//!     }
//!     // ... spawn daemon ...
//! }
//! ```
//!
//! ```rust,ignore
//! // In hive-lifecycle (queen → hive)
//! use auto_update::AutoUpdater;
//!
//! pub async fn spawn_hive(config: &Config) -> Result<Child> {
//!     if config.auto_update_hive {
//!         AutoUpdater::new("rbee-hive", "bin/20_rbee_hive")?
//!             .ensure_built()
//!             .await?;
//!     }
//!     // ... spawn daemon ...
//! }
//! ```
//!
//! ```rust,ignore
//! // In worker-lifecycle (hive → worker)
//! use auto_update::AutoUpdater;
//!
//! pub async fn spawn_worker(config: &Config) -> Result<Child> {
//!     if config.auto_update_worker {
//!         AutoUpdater::new("llm-worker-rbee", "bin/30_llm_worker_rbee")?
//!             .ensure_built()
//!             .await?;
//!     }
//!     // ... spawn daemon ...
//! }
//! ```

use anyhow::{anyhow, Context, Result};
use cargo_toml::Manifest;
use observability_narration_core::NarrationFactory;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;
use walkdir::WalkDir;

// Narration factory for auto-update operations
const NARRATE: NarrationFactory = NarrationFactory::new("auto-upd");

/// Auto-updater for rbee binaries with full dependency tracking
///
/// # Example
///
/// ```rust,ignore
/// use auto_update::AutoUpdater;
///
/// // Create updater for queen-rbee
/// let updater = AutoUpdater::new("queen-rbee", "bin/10_queen_rbee")?;
///
/// // Check if rebuild needed (checks ALL dependencies)
/// if updater.needs_rebuild()? {
///     println!("Rebuilding queen-rbee...");
///     updater.rebuild()?;
/// }
///
/// // Get binary path
/// let binary_path = updater.find_binary()?;
/// ```
pub struct AutoUpdater {
    /// Binary name (e.g., "queen-rbee")
    binary_name: String,

    /// Source directory relative to workspace root (e.g., "bin/10_queen_rbee")
    source_dir: PathBuf,

    /// Workspace root (auto-detected)
    workspace_root: PathBuf,

    /// Cached dependency graph
    dependencies: Vec<PathBuf>,
}

impl AutoUpdater {
    /// Create a new auto-updater
    ///
    /// # Arguments
    /// * `binary_name` - Binary name (e.g., "queen-rbee")
    /// * `source_dir` - Source directory relative to workspace root (e.g., "bin/10_queen_rbee")
    ///
    /// # Returns
    /// * `Ok(AutoUpdater)` - Ready to check/rebuild
    /// * `Err` - Failed to find workspace root or parse dependencies
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let updater = AutoUpdater::new("queen-rbee", "bin/10_queen_rbee")?;
    /// ```
    pub fn new(binary_name: impl Into<String>, source_dir: impl Into<PathBuf>) -> Result<Self> {
        let binary_name = binary_name.into();
        let source_dir = source_dir.into();

        // Auto-detect workspace root
        let workspace_root = Self::find_workspace_root()?;

        NARRATE
            .action("init")
            .context(&binary_name)
            .human("🔨 Initializing auto-updater for {}")
            .emit();

        // Parse dependencies from Cargo.toml
        let dependencies = Self::parse_dependencies(&workspace_root, &source_dir)?;

        NARRATE
            .action("deps_parsed")
            .context(format!("{} dependencies", dependencies.len()))
            .human("📦 Found {}")
            .emit();

        Ok(Self { binary_name, source_dir, workspace_root, dependencies })
    }

    /// Check if binary needs rebuilding
    ///
    /// Checks:
    /// 1. Binary exists in target/debug or target/release
    /// 2. Binary mtime vs source directory mtime
    /// 3. Binary mtime vs ALL dependency mtimes (recursive)
    ///
    /// # Returns
    /// * `Ok(true)` - Rebuild needed
    /// * `Ok(false)` - Binary is up-to-date
    /// * `Err` - Failed to check
    pub fn needs_rebuild(&self) -> Result<bool> {
        NARRATE
            .action("check_rebuild")
            .context(&self.binary_name)
            .human("🔍 Checking if {} needs rebuild")
            .emit();

        // Find binary (debug or release)
        let binary_path = match self.find_binary_path() {
            Ok(path) => path,
            Err(_) => {
                NARRATE
                    .action("check_rebuild")
                    .context(&self.binary_name)
                    .human("⚠️  Binary {} not found, rebuild needed")
                    .emit();
                return Ok(true);
            }
        };

        let binary_time =
            std::fs::metadata(&binary_path)?.modified().context("Failed to get binary mtime")?;

        // Check binary's own source directory
        let source_path = self.workspace_root.join(&self.source_dir);
        if Self::is_dir_newer(&source_path, binary_time)? {
            NARRATE
                .action("check_rebuild")
                .context(self.source_dir.display().to_string())
                .human("🔨 Source directory {} changed, rebuild needed")
                .emit();
            return Ok(true);
        }

        // Check ALL dependencies
        for dep_path in &self.dependencies {
            let full_path = self.workspace_root.join(dep_path);
            if Self::is_dir_newer(&full_path, binary_time)? {
                NARRATE
                    .action("check_rebuild")
                    .context(dep_path.display().to_string())
                    .human("🔨 Dependency {} changed, rebuild needed")
                    .emit();
                return Ok(true);
            }
        }

        NARRATE
            .action("check_rebuild")
            .context(&self.binary_name)
            .human("✅ Binary {} is up-to-date")
            .emit();

        Ok(false)
    }

    /// Rebuild the binary
    ///
    /// Runs `cargo build --bin <binary_name>` in workspace root.
    ///
    /// # Returns
    /// * `Ok(())` - Build succeeded
    /// * `Err` - Build failed
    pub fn rebuild(&self) -> Result<()> {
        NARRATE.action("rebuild").context(&self.binary_name).human("🔨 Rebuilding {}...").emit();

        let start_time = std::time::Instant::now();

        let status = Command::new("cargo")
            .arg("build")
            .arg("--bin")
            .arg(&self.binary_name)
            .current_dir(&self.workspace_root)
            .status()
            .context("Failed to run cargo build")?;

        if !status.success() {
            NARRATE
                .action("rebuild")
                .context(&self.binary_name)
                .human("❌ Failed to rebuild {}")
                .error_kind("build_failed")
                .emit();
            anyhow::bail!("Build failed for {}", self.binary_name);
        }

        let elapsed_ms = start_time.elapsed().as_millis() as u64;

        NARRATE
            .action("rebuild")
            .context(&self.binary_name)
            .human("✅ Rebuilt {} successfully")
            .duration_ms(elapsed_ms)
            .emit();

        Ok(())
    }

    /// Find binary in target directory
    ///
    /// Searches:
    /// 1. `target/debug/<binary_name>`
    /// 2. `target/release/<binary_name>`
    ///
    /// # Returns
    /// * `Ok(PathBuf)` - Path to binary
    /// * `Err` - Binary not found
    pub fn find_binary(&self) -> Result<PathBuf> {
        self.find_binary_path()
    }

    /// Ensure binary is built (check + rebuild if needed)
    ///
    /// Convenience method that combines `needs_rebuild()` and `rebuild()`.
    ///
    /// # Returns
    /// * `Ok(PathBuf)` - Path to up-to-date binary
    /// * `Err` - Failed to rebuild or find binary
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let binary_path = AutoUpdater::new("queen-rbee", "bin/10_queen_rbee")?
    ///     .ensure_built()
    ///     .await?;
    ///
    /// // Spawn daemon with up-to-date binary
    /// Command::new(&binary_path).spawn()?;
    /// ```
    pub async fn ensure_built(self) -> Result<PathBuf> {
        if self.needs_rebuild()? {
            self.rebuild()?;
        }

        self.find_binary()
    }

    // ============================================================================
    // PRIVATE HELPERS
    // ============================================================================

    /// Find workspace root by walking up directory tree
    fn find_workspace_root() -> Result<PathBuf> {
        let mut current = std::env::current_dir().context("Failed to get current directory")?;

        loop {
            let cargo_toml = current.join("Cargo.toml");
            if cargo_toml.exists() {
                let contents =
                    std::fs::read_to_string(&cargo_toml).context("Failed to read Cargo.toml")?;
                if contents.contains("[workspace]") {
                    return Ok(current);
                }
            }

            current =
                current.parent().ok_or_else(|| anyhow!("Workspace root not found"))?.to_path_buf();
        }
    }

    /// Parse all local path dependencies from Cargo.toml (recursive)
    fn parse_dependencies(workspace_root: &Path, source_dir: &Path) -> Result<Vec<PathBuf>> {
        let mut all_deps = Vec::new();
        let mut visited = HashSet::new();

        Self::collect_deps_recursive(workspace_root, source_dir, &mut all_deps, &mut visited)?;

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
    fn collect_deps_recursive(
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

        let manifest = Manifest::from_path(&cargo_toml)
            .with_context(|| format!("Failed to parse {}", cargo_toml.display()))?;

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

                    // Recursively check this dependency's dependencies
                    Self::collect_deps_recursive(workspace_root, &dep_relative, all_deps, visited)?;
                }
            }
        }

        Ok(())
    }

    /// Check if any file in directory is newer than reference time
    fn is_dir_newer(dir: &Path, reference_time: SystemTime) -> Result<bool> {
        if !dir.exists() {
            return Ok(false);
        }

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
                    let meta = std::fs::metadata(path)?;
                    if let Ok(modified) = meta.modified() {
                        if modified > reference_time {
                            return Ok(true);
                        }
                    }
                }
            }
        }

        Ok(false)
    }

    /// Find binary path (debug or release)
    fn find_binary_path(&self) -> Result<PathBuf> {
        // Try debug first (development mode)
        let debug_path = self.workspace_root.join("target/debug").join(&self.binary_name);
        if debug_path.exists() {
            return Ok(debug_path);
        }

        // Try release
        let release_path = self.workspace_root.join("target/release").join(&self.binary_name);
        if release_path.exists() {
            return Ok(release_path);
        }

        anyhow::bail!("Binary '{}' not found in target/debug or target/release", self.binary_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_workspace_root() {
        let root = AutoUpdater::find_workspace_root();
        assert!(root.is_ok());
        let root = root.unwrap();
        assert!(root.join("Cargo.toml").exists());
    }

    #[test]
    fn test_parse_dependencies() {
        let root = AutoUpdater::find_workspace_root().unwrap();
        let deps =
            AutoUpdater::parse_dependencies(&root, &PathBuf::from("bin/00_rbee_keeper")).unwrap();

        // rbee-keeper should have dependencies
        assert!(!deps.is_empty());

        // Should include daemon-lifecycle
        assert!(deps.iter().any(|d| d.to_string_lossy().contains("daemon-lifecycle")));

        // Should include narration-core
        assert!(deps.iter().any(|d| d.to_string_lossy().contains("narration-core")));
    }

    #[test]
    fn test_new_rbee_keeper() {
        let updater = AutoUpdater::new("rbee-keeper", "bin/00_rbee_keeper");
        assert!(updater.is_ok());

        let updater = updater.unwrap();
        assert_eq!(updater.binary_name, "rbee-keeper");
        assert!(!updater.dependencies.is_empty());
    }

    #[test]
    fn test_find_binary() {
        // This test assumes rbee-keeper is built
        let updater = AutoUpdater::new("rbee-keeper", "bin/00_rbee_keeper").unwrap();
        let binary = updater.find_binary();

        // May fail if binary not built, that's ok
        if let Ok(path) = binary {
            assert!(path.exists());
            assert!(path.to_string_lossy().contains("rbee-keeper"));
        }
    }
}
