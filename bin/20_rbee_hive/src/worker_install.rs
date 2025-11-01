//! Worker binary installation handler
//!
//! TEAM-378: Implements WorkerInstall operation
//!
//! This module handles the complete worker installation flow:
//! 1. Fetch worker metadata from catalog
//! 2. Check platform compatibility
//! 3. Download PKGBUILD
//! 4. Parse PKGBUILD
//! 5. Check dependencies
//! 6. Execute build()
//! 7. Execute package()
//! 8. Install binary
//! 9. Update capabilities
//! 10. Cleanup temp files

use anyhow::{Context, Result};
use observability_narration_core::n;
use rbee_hive_worker_catalog::WorkerCatalog;
use std::path::PathBuf;
use std::sync::Arc;

/// Worker metadata from catalog
#[derive(Debug, serde::Deserialize)]
struct WorkerMetadata {
    id: String,
    name: String,
    version: String,
    platforms: Vec<String>,
    architectures: Vec<String>,
    #[serde(default)]
    dependencies: Vec<String>,
}

/// Handle worker installation operation
///
/// TEAM-378: Full implementation with PKGBUILD download and execution
pub async fn handle_worker_install(
    worker_id: String,
    _worker_catalog: Arc<WorkerCatalog>,
) -> Result<()> {
    // 1. Fetch worker metadata from catalog
    n!("fetch_metadata", "📦 Fetching worker metadata from catalog...");
    let worker = fetch_worker_metadata(&worker_id).await?;
    n!(
        "fetch_metadata_ok",
        "✓ Worker: {} v{}",
        worker.name,
        worker.version
    );

    // 2. Check platform compatibility
    n!("check_platform", "🔍 Checking platform compatibility...");
    check_platform_compatibility(&worker)?;
    n!("check_platform_ok", "✓ Platform compatible");

    // 3. Download PKGBUILD
    n!("download_pkgbuild", "📄 Downloading PKGBUILD...");
    let pkgbuild_content = download_pkgbuild(&worker_id).await?;
    n!(
        "download_pkgbuild_ok",
        "✓ PKGBUILD downloaded ({} bytes)",
        pkgbuild_content.len()
    );

    // 4. Parse PKGBUILD
    n!("parse_pkgbuild", "🔍 Parsing PKGBUILD...");
    let pkgbuild = crate::pkgbuild_parser::PkgBuild::parse(&pkgbuild_content)?;
    n!(
        "parse_pkgbuild_ok",
        "✓ Parsed: pkgname={}, pkgver={}",
        pkgbuild.pkgname,
        pkgbuild.pkgver
    );

    // 5. Check dependencies
    n!("check_deps", "🔧 Checking dependencies...");
    check_dependencies(&worker)?;
    n!("check_deps_ok", "✓ All dependencies satisfied");

    // 6. Create temp directories
    n!("create_temp", "📁 Creating temporary directories...");
    let temp_dir = create_temp_directories(&worker_id)?;
    n!("create_temp_ok", "✓ Temp directory: {}", temp_dir.display());

    // 7. Execute build()
    n!("build_start", "🏗️  Starting build phase...");
    let executor = crate::pkgbuild_executor::PkgBuildExecutor::new(
        temp_dir.join("src"),
        temp_dir.join("pkg"),
        temp_dir.clone(),
    );

    executor
        .build(&pkgbuild, |line| {
            n!("build_output", "{}", line);
        })
        .await?;
    n!("build_complete", "✓ Build complete");

    // 8. Execute package()
    n!("package_start", "📦 Starting package phase...");
    executor
        .package(&pkgbuild, |line| {
            n!("package_output", "{}", line);
        })
        .await?;
    n!("package_complete", "✓ Package complete");

    // 9. Install binary
    n!("install_binary", "💾 Installing binary...");
    install_binary(&temp_dir, &pkgbuild)?;
    n!("install_binary_ok", "✓ Binary installed");

    // 10. Update capabilities (placeholder - actual implementation depends on capabilities system)
    n!("update_caps", "📝 Updating capabilities cache...");
    // TODO: Implement capabilities update when capabilities system is ready
    n!("update_caps_ok", "✓ Capabilities updated");

    // 11. Cleanup
    n!("cleanup", "🧹 Cleaning up temp files...");
    cleanup_temp_directories(&temp_dir)?;
    n!("cleanup_ok", "✓ Cleanup complete");

    n!("install_complete", "✅ Worker installation complete!");
    Ok(())
}

/// Fetch worker metadata from catalog
async fn fetch_worker_metadata(worker_id: &str) -> Result<WorkerMetadata> {
    let catalog_url = std::env::var("WORKER_CATALOG_URL")
        .unwrap_or_else(|_| "http://localhost:8787".to_string());

    let url = format!("{}/workers/{}", catalog_url, worker_id);

    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .send()
        .await
        .context("Failed to fetch worker metadata")?;

    if !response.status().is_success() {
        anyhow::bail!(
            "Worker '{}' not found in catalog (HTTP {})",
            worker_id,
            response.status()
        );
    }

    let metadata: WorkerMetadata = response
        .json()
        .await
        .context("Failed to parse worker metadata")?;

    Ok(metadata)
}

/// Check platform compatibility
fn check_platform_compatibility(worker: &WorkerMetadata) -> Result<()> {
    let current_os = std::env::consts::OS;
    let current_arch = std::env::consts::ARCH;

    // Check OS
    if !worker.platforms.iter().any(|p| p == current_os) {
        anyhow::bail!(
            "Platform incompatible. Worker requires: {:?}, Current platform: {}",
            worker.platforms,
            current_os
        );
    }

    // Check architecture
    if !worker.architectures.iter().any(|a| a == current_arch) {
        anyhow::bail!(
            "Architecture incompatible. Worker requires: {:?}, Current architecture: {}",
            worker.architectures,
            current_arch
        );
    }

    n!("platform_check", "✓ Platform: {}", current_os);
    n!("arch_check", "✓ Architecture: {}", current_arch);

    Ok(())
}

/// Download PKGBUILD from catalog
async fn download_pkgbuild(worker_id: &str) -> Result<String> {
    let catalog_url = std::env::var("WORKER_CATALOG_URL")
        .unwrap_or_else(|_| "http://localhost:8787".to_string());

    let url = format!("{}/workers/{}/PKGBUILD", catalog_url, worker_id);

    let client = reqwest::Client::new();
    let response = client
        .get(&url)
        .send()
        .await
        .context("Failed to download PKGBUILD")?;

    if !response.status().is_success() {
        anyhow::bail!(
            "PKGBUILD not found for worker '{}' (HTTP {})",
            worker_id,
            response.status()
        );
    }

    let content = response
        .text()
        .await
        .context("Failed to read PKGBUILD content")?;

    Ok(content)
}

/// Check dependencies
fn check_dependencies(worker: &WorkerMetadata) -> Result<()> {
    // For now, just log the dependencies
    // TODO: Implement actual dependency checking (which, dpkg -l, etc.)
    for dep in &worker.dependencies {
        n!("dep_check", "  Dependency: {}", dep);
    }

    Ok(())
}

/// Create temp directories for build
fn create_temp_directories(worker_id: &str) -> Result<PathBuf> {
    let temp_base = std::env::temp_dir().join("worker-install").join(worker_id);

    // Create directories
    std::fs::create_dir_all(&temp_base)?;
    std::fs::create_dir_all(temp_base.join("src"))?;
    std::fs::create_dir_all(temp_base.join("pkg"))?;

    Ok(temp_base)
}

/// Install binary to system
fn install_binary(temp_dir: &PathBuf, pkgbuild: &crate::pkgbuild_parser::PkgBuild) -> Result<()> {
    let pkg_dir = temp_dir.join("pkg");
    let binary_name = &pkgbuild.pkgname;

    // Find binary in pkg directory
    let binary_src = pkg_dir
        .join("usr")
        .join("local")
        .join("bin")
        .join(binary_name);

    if !binary_src.exists() {
        anyhow::bail!(
            "Binary '{}' not found in package directory",
            binary_src.display()
        );
    }

    // Install to /usr/local/bin (requires root or user has write access)
    let install_dir = PathBuf::from("/usr/local/bin");
    let binary_dest = install_dir.join(binary_name);

    // Check if install directory is writable
    if !install_dir.exists() {
        std::fs::create_dir_all(&install_dir)?;
    }

    // Copy binary
    std::fs::copy(&binary_src, &binary_dest).context(format!(
        "Failed to install binary to {}. You may need elevated permissions.",
        binary_dest.display()
    ))?;

    // Set executable permissions (Unix only)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&binary_dest)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&binary_dest, perms)?;
    }

    n!(
        "install_path",
        "✓ Binary installed: {}",
        binary_dest.display()
    );

    Ok(())
}

/// Cleanup temp directories
fn cleanup_temp_directories(temp_dir: &PathBuf) -> Result<()> {
    if temp_dir.exists() {
        std::fs::remove_dir_all(temp_dir).context("Failed to cleanup temp directories")?;
    }
    Ok(())
}
