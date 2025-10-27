//! Binary finding utilities
//!
//! TEAM-259: Extracted from lib.rs for better organization
//! TEAM-329: Removed spawn() method - logic inlined into start.rs
//! TEAM-329: Renamed manager.rs → find.rs (accurate naming)
//!
//! Provides find_binary() for locating daemon binaries.

use anyhow::Result;
use observability_narration_core::n;
use std::path::PathBuf;

// TEAM-329: DELETED DaemonManager struct - RULE ZERO violation (entropy)
//
// PROBLEM: DaemonManager was a wrapper around spawn() which we deleted.
// The only remaining useful method is find_binary(), which doesn't need a struct.
//
// SOLUTION: Made find_binary() a standalone function. All spawn logic is now
// directly in start.rs:start_daemon() where it belongs.
//
// HISTORICAL CONTEXT:
// - TEAM-259: Created DaemonManager for spawn() + auto-update
// - TEAM-164: Fixed pipe inheritance bug (Stdio::null())
// - TEAM-189: SSH agent propagation
// - TEAM-329: Eliminated entropy by inlining spawn() and deleting struct
//
// If worker-lifecycle needs spawning, it should use start_daemon() or implement
// its own logic. Don't create wrapper structs for single functions.

/// Find a binary (installed or development)
///
/// TEAM-320: Checks installed location first, then falls back to development builds
/// TEAM-329: Converted from DaemonManager method to standalone function
///
/// Search order:
/// 1. `~/.local/bin/{name}` (installed)
/// 2. `target/debug/{name}` (development)
/// 3. `target/release/{name}` (development)
///
/// # Arguments
/// * `name` - Binary name (e.g., "queen-rbee")
///
/// # Returns
/// Path to the binary if found
///
/// # Example
/// ```rust,no_run
/// use daemon_lifecycle::find_binary;
///
/// # fn example() -> anyhow::Result<()> {
/// let binary = find_binary("queen-rbee")?;
/// println!("Found at: {}", binary.display());
/// # Ok(())
/// # }
/// ```
pub fn find_binary(name: &str) -> Result<PathBuf> {
        // Try installed location first
        // TEAM-328: Use centralized path function
        if let Ok(installed_path) = crate::utils::paths::get_install_path(name) {
            if installed_path.exists() {
                n!("find_binary", "Found installed binary '{}' at: {}", name, installed_path.display());
                return Ok(installed_path);
            }
        }
        
        // Fall back to development builds
        // TEAM-255: Find workspace root by looking for Cargo.toml
        let mut current = std::env::current_dir()?;
        let workspace_root = loop {
            if current.join("Cargo.toml").exists()
                && (current.join("xtask").exists() || current.join("bin").exists())
            {
                break current;
            }
            if let Some(parent) = current.parent() {
                current = parent.to_path_buf();
            } else {
                // Fallback to current dir if workspace root not found
                break std::env::current_dir()?;
            }
        };

        // Try debug first (development mode)
        let debug_path = workspace_root.join("target/debug").join(name);
        if debug_path.exists() {
            n!("find_binary", "Found binary '{}' at: {}", name, debug_path.display());
            return Ok(debug_path);
        }

        // Try release
        let release_path = workspace_root.join("target/release").join(name);
        if release_path.exists() {
            n!("find_binary", "Found binary '{}' at: {}", name, release_path.display());
            return Ok(release_path);
        }

        n!("find_binary", "Binary '{}' not found in target/debug or target/release", name);
        anyhow::bail!("Binary '{}' not found in target/debug or target/release", name)
}
