// TEAM-309: Extracted AutoUpdater struct and public API
//! Core AutoUpdater struct and public methods

use anyhow::{Context, Result};
use observability_narration_core::n;
use observability_narration_macros::narrate_fn;
use std::path::PathBuf;

use crate::binary::BinaryFinder;
use crate::checker::RebuildChecker;
use crate::dependencies::DependencyParser;
use crate::rebuild::Rebuilder;
use crate::workspace::WorkspaceFinder;

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
    pub(crate) binary_name: String,

    /// Source directory relative to workspace root (e.g., "bin/10_queen_rbee")
    pub(crate) source_dir: PathBuf,

    /// Workspace root (auto-detected)
    pub(crate) workspace_root: PathBuf,

    /// Cached dependency graph
    pub(crate) dependencies: Vec<PathBuf>,
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

        // TEAM-309: Added narration
        n!("init", "🔨 Initializing auto-updater for {}", binary_name);

        // Auto-detect workspace root
        let workspace_root = WorkspaceFinder::find()?;

        // Parse dependencies from Cargo.toml
        let dependencies = DependencyParser::parse(&workspace_root, &source_dir)?;

        n!("deps_parsed", "📦 Found {} dependencies", dependencies.len());

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
        use crate::checker::RebuildChecker;
        
        RebuildChecker::check(self)
    }

    /// Rebuild the binary
    ///
    /// Runs `cargo build --bin <binary_name>` in workspace root.
    ///
    /// # Returns
    /// * `Ok(())` - Build succeeded
    /// * `Err` - Build failed
    pub fn rebuild(&self) -> Result<()> {
        use crate::rebuild::Rebuilder;
        Rebuilder::rebuild(self)
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
        use crate::binary::BinaryFinder;
        
        BinaryFinder::find_path(self)
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
    #[narrate_fn]
    pub async fn ensure_built(self) -> Result<PathBuf> {
        // TEAM-309: Added narration for ensure_built flow
        n!("ensure_built", "🔍 Ensuring {} is built", self.binary_name);
        
        if self.needs_rebuild()? {
            self.rebuild()?;
        }

        let binary_path = self.find_binary()?;
        
        n!("ensure_built", "✅ Binary {} ready at {}", self.binary_name, binary_path.display());
        
        Ok(binary_path)
    }
}
