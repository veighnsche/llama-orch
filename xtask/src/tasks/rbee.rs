// Created by: TEAM-162
// Smart wrapper for rbee-keeper: checks if build is stale, rebuilds if needed, then forwards command

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

const RBEE_KEEPER_BIN: &str = "bin/00_rbee_keeper";
const TARGET_BINARY: &str = "target/debug/rbee-keeper";

/// Check if rbee-keeper binary needs rebuilding
fn needs_rebuild(workspace_root: &Path) -> Result<bool> {
    let binary_path = workspace_root.join(TARGET_BINARY);

    // If binary doesn't exist, definitely need to build
    if !binary_path.exists() {
        return Ok(true);
    }

    // Get binary modification time
    let binary_meta = std::fs::metadata(&binary_path).context("Failed to get binary metadata")?;
    let binary_time = binary_meta.modified().context("Failed to get binary modification time")?;

    // Check if any source files in rbee-keeper are newer
    let keeper_dir = workspace_root.join(RBEE_KEEPER_BIN);
    let needs_rebuild = check_dir_newer(&keeper_dir, binary_time)?;

    Ok(needs_rebuild)
}

/// Recursively check if any .rs or Cargo.toml files in dir are newer than reference_time
fn check_dir_newer(dir: &Path, reference_time: SystemTime) -> Result<bool> {
    if !dir.exists() {
        return Ok(false);
    }

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        // Skip target directories
        if path.is_dir() && path.file_name().map(|n| n == "target").unwrap_or(false) {
            continue;
        }

        if path.is_dir() {
            if check_dir_newer(&path, reference_time)? {
                return Ok(true);
            }
        } else if let Some(ext) = path.extension() {
            if ext == "rs" || path.file_name().map(|n| n == "Cargo.toml").unwrap_or(false) {
                let meta = std::fs::metadata(&path)?;
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

/// Build rbee-keeper binary
fn build_rbee_keeper(workspace_root: &Path) -> Result<()> {
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
    } else {
        println!("✅ rbee-keeper is up-to-date\n");
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
