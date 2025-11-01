// Created by: TEAM-162
// Smart wrapper for rbee-keeper: checks if build is stale, rebuilds if needed, then forwards command
// TEAM-193: Updated to use auto-update crate for dependency-aware rebuilds
// TEAM-309: Added narration context for auto-update visibility
// TEAM-334: Used by desktop entry (./rbee → xtask rbee → rbee-keeper)
//
// TEAM-334 NOTE: Desktop entry integration
// - Called by: ./rbee script (root of repo)
// - Desktop entry: ~/.local/share/applications/rbee-dev.desktop
// - When called with no args, launches rbee-keeper GUI
// - Status: ⚠️ NOT WORKING YET from desktop entry (process starts but GUI doesn't show)
// - Works fine when called directly: ./rbee
// - See: DESKTOP_ENTRY.md for debugging info

use anyhow::{Context, Result};
use auto_update::AutoUpdater;
use std::path::PathBuf;
use std::process::Command;

const RBEE_KEEPER_BIN: &str = "bin/00_rbee_keeper";
const TARGET_BINARY: &str = "target/debug/rbee-keeper";

/// Check if rbee-keeper binary needs rebuilding
/// TEAM-193: Now uses AutoUpdater to check ALL dependencies (including shared crates)
fn needs_rebuild(_workspace_root: &PathBuf) -> Result<bool> {
    // TEAM-193: Use auto-update crate for dependency-aware rebuild detection
    // This checks:
    // 1. bin/00_rbee_keeper/ (source)
    // 2. ALL Cargo.toml dependencies (daemon-lifecycle, narration-core, etc.)
    // 3. Transitive dependencies (dependencies of dependencies)
    let updater = AutoUpdater::new("rbee-keeper", RBEE_KEEPER_BIN)?;
    updater.needs_rebuild()
}

// TEAM-193: Removed check_dir_newer() - now handled by AutoUpdater
// AutoUpdater parses Cargo.toml and checks ALL dependencies recursively

/// Build rbee-keeper binary
fn build_rbee_keeper(workspace_root: &PathBuf) -> Result<()> {
    println!("🔨 Building rbee-keeper...");

    let status = Command::new("cargo")
        .arg("build")
        .arg("--bin")
        .arg("rbee-keeper")
        .current_dir(workspace_root)
        .status()
        .context("Failed to run cargo build")?;

    if !status.success() {
        anyhow::bail!("Failed to build rbee-keeper");
    }

    println!("✅ Build complete\n");
    Ok(())
}

/// Main entry point: check build status, rebuild if needed, then forward to rbee-keeper
pub fn run_rbee_keeper(args: Vec<String>) -> Result<()> {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .context("Failed to get workspace root")?
        .to_path_buf();

    // Check if rebuild is needed
    if needs_rebuild(&workspace_root)? {
        build_rbee_keeper(&workspace_root)?;
    }

    // Forward command to rbee-keeper
    let binary_path = workspace_root.join(TARGET_BINARY);

    let status = Command::new(&binary_path)
        .args(&args)
        .current_dir(&workspace_root)
        .status()
        .context("Failed to execute rbee-keeper")?;

    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }

    Ok(())
}
