// TEAM-309: Extracted AutoUpdater struct and public API
//! Core AutoUpdater struct and public methods

use anyhow::Result;
use observability_narration_core::{n, with_narration_context, NarrationContext};
use std::path::PathBuf;
use std::time::Instant;

use crate::dependencies::DependencyParser;
use crate::workspace::WorkspaceFinder;

/// Auto-updater for rbee binaries with full dependency tracking
///
/// TEAM-311: Uses V2 narration format with six phases
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

    /// Optional job_id for narration context
    pub(crate) job_id: Option<String>,
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
    ///
    pub fn new(binary_name: impl Into<String>, source_dir: impl Into<PathBuf>) -> Result<Self> {
        let binary_name = binary_name.into();
        let source_dir = source_dir.into();

        // TEAM-311: Phase 1 - Init
        let start = Instant::now();
        n!("phase_init", "🚧 Initializing auto-updater for {}", binary_name);

        let mode = if cfg!(debug_assertions) { "debug" } else { "release" };
        n!("init", "Mode: {} · Binary: {} · Source: {}", mode, binary_name, source_dir.display());

        let elapsed = start.elapsed().as_millis();
        n!("summary", "✅ Init ok · {}ms", elapsed);

        // TEAM-311: Phase 2 - Workspace
        let workspace_root = WorkspaceFinder::find()?;

        // TEAM-311: Phase 3 - Dependencies
        let dependencies = DependencyParser::parse(&workspace_root, &source_dir)?;

        Ok(Self { binary_name, source_dir, workspace_root, dependencies, job_id: None })
    }

    /// Set job_id for narration context
    ///
    /// TEAM-311: Enables context propagation for all narrations
    pub fn with_job_id(mut self, job_id: impl Into<String>) -> Self {
        self.job_id = Some(job_id.into());
        self
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
    /// TEAM-311: Uses V2 narration format with context propagation
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
    ///     .with_job_id("job-123")
    ///     .ensure_built()
    ///     .await?;
    ///
    /// // Spawn daemon with up-to-date binary
    /// Command::new(&binary_path).spawn()?;
    /// ```
    pub async fn ensure_built(self) -> Result<PathBuf> {
        // TEAM-311: Wrap in context if job_id provided
        let ctx = self.job_id.as_ref().map(|jid| NarrationContext::new().with_job_id(jid));

        let impl_fn = async {
            // All phases run with context
            if self.needs_rebuild()? {
                self.rebuild()?;
            }

            let binary_path = self.find_binary()?;
            Ok(binary_path)
        };

        if let Some(ctx) = ctx {
            with_narration_context(ctx, impl_fn).await
        } else {
            impl_fn.await
        }
    }
}
