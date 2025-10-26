// TEAM-193: Created by TEAM-193
// TEAM-309: Refactored into modular structure
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
//! # Narration
//!
//! This crate uses the ultra-concise `n!()` macro for narration with actor set to "auto-update":
//!
//! - **Initialization** - When AutoUpdater is created
//! - **Workspace discovery** - When finding workspace root
//! - **Dependency parsing** - When parsing Cargo.toml files
//! - **Rebuild checking** - When checking if rebuild is needed
//! - **File scanning** - When checking file timestamps
//! - **Binary search** - When looking for binary in target directories
//! - **Rebuilding** - When running cargo build
//!
//! TEAM-309: Uses #[with_actor("auto-update")] macro to set actor for all narration
//!
//! All narration supports 3 modes (human/cute/story) and can be configured at runtime.

mod binary;
mod checker;
mod dependencies;
mod rebuild;
mod updater;
mod workspace;

// Re-export main struct
pub use updater::AutoUpdater;
