// TEAM-309: Extracted workspace root finding logic
//! Workspace root discovery

use anyhow::{anyhow, Context, Result};
use observability_narration_core::n;
use std::path::PathBuf;
use std::time::Instant;

/// Workspace root finder
pub struct WorkspaceFinder;

impl WorkspaceFinder {
    /// Find workspace root by walking up directory tree
    ///
    /// # Returns
    /// * `Ok(PathBuf)` - Workspace root directory
    /// * `Err` - Workspace root not found
    pub fn find() -> Result<PathBuf> {
        // TEAM-311: Phase 2 - Workspace
        let start = Instant::now();
        n!("phase_workspace", "🧭 Workspace detection");
        
        let mut current = std::env::current_dir().context("Failed to get current directory")?;
        let mut depth = 0;

        loop {
            let cargo_toml = current.join("Cargo.toml");
            
            if cargo_toml.exists() {
                let contents =
                    std::fs::read_to_string(&cargo_toml).context("Failed to read Cargo.toml")?;
                    
                if contents.contains("[workspace]") {
                    n!("find_workspace", "Searching for workspace root · depth: {}", depth);
                    n!("workspace_found", "Workspace: {}", current.display());
                    
                    let elapsed = start.elapsed().as_millis();
                    n!("summary", "✅ Workspace ok · {}ms", elapsed);
                    
                    return Ok(current);
                }
            }

            depth += 1;
            current =
                current.parent().ok_or_else(|| anyhow!("Workspace root not found"))?.to_path_buf();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_workspace_root() {
        let root = WorkspaceFinder::find();
        assert!(root.is_ok());
        let root = root.unwrap();
        assert!(root.join("Cargo.toml").exists());
    }
}
