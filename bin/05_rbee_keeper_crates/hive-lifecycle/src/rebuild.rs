//! Hive rebuild operation
//!
//! TEAM-314: Added for parity with queen-lifecycle rebuild

use anyhow::Result;
use observability_narration_core::n;
use ssh_config::SshClient; // TEAM-314: Use shared SSH client

use crate::{DEFAULT_INSTALL_DIR, DEFAULT_BUILD_DIR};

/// Rebuild rbee-hive from source (update operation)
///
/// TEAM-314: Mirrors queen rebuild functionality
/// DEFAULT = build locally + upload, OPTIONAL = build on-site
///
/// For localhost: Runs `cargo build --release --bin rbee-hive` locally
/// For remote (default): Build locally, upload binary
/// For remote (--build-remote): Build on-site via SSH
///
/// # Arguments
/// * `host` - Host to rebuild on ("localhost" for local, SSH alias for remote)
/// * `build_remote` - If true, build on remote host (requires git). Default: build locally + upload
///
/// # Returns
/// * `Ok(())` - Build successful
/// * `Err` - Build failed or hive is running
pub async fn rebuild_hive(host: &str, build_remote: bool) -> Result<()> {
    n!("rebuild_start", "🔄 Updating rbee-hive (rebuilding from source)...");
    
    // Check if localhost (direct rebuild) or remote (SSH rebuild)
    if host == "localhost" || host == "127.0.0.1" {
        rebuild_hive_local().await
    } else {
        rebuild_hive_remote(host, build_remote).await
    }
}

/// Rebuild rbee-hive locally (no SSH)
///
/// TEAM-314: Mirrors queen rebuild for localhost
async fn rebuild_hive_local() -> Result<()> {
    // TEAM-314: Check if hive is running (same as queen)
    let hive_url = "http://localhost:7835";
    let is_running = daemon_lifecycle::health::is_daemon_healthy(
        hive_url,
        None, // Use default /health endpoint
        Some(std::time::Duration::from_secs(2)),
    )
    .await;
    
    if is_running {
        n!("daemon_still_running", "⚠️  Hive is currently running. Stop it first.");
        anyhow::bail!("Hive is still running. Use 'rbee hive stop' first.");
    }

    // Execute build
    n!("build_start", "⏳ Running cargo build (this may take a few minutes)...");

    let mut cmd = std::process::Command::new("cargo");
    cmd.arg("build")
        .arg("--release")
        .arg("--bin")
        .arg("rbee-hive");

    let output = cmd.output()?;

    if output.status.success() {
        n!("build_success", "✅ Build successful!");

        // Show binary location
        let binary_path = "target/release/rbee-hive";
        n!("binary_location", "📦 Binary available at: {}", binary_path);
        n!("restart_hint", "💡 Restart hive to use the new binary");
        
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        n!("build_failed", "❌ Build failed: {}", stderr);
        anyhow::bail!("Build failed");
    }
}

/// Rebuild rbee-hive remotely via SSH
///
/// TEAM-314: DEFAULT = build locally + upload (fast, no git)
///           OPTIONAL = build on-site (slower, requires git)
async fn rebuild_hive_remote(host: &str, build_remote: bool) -> Result<()> {
    n!("rebuild_remote", "🔄 Rebuilding rbee-hive on '{}' via SSH...", host);
    
    let client = SshClient::connect(host).await?;
    
    // TEAM-314: Check if hive is running
    let is_running = client
        .execute("pgrep -f rbee-hive")
        .await
        .is_ok();
    
    if is_running {
        n!("daemon_still_running", "⚠️  Hive is currently running on '{}'. Stop it first.", host);
        anyhow::bail!("Hive is still running on '{}'. Use 'rbee hive stop -a {}' first.", host, host);
    }
    
    // TEAM-314: If build_remote is true, use on-site build
    if build_remote {
        return rebuild_hive_remote_onsite(host, &client).await;
    }
    
    // TEAM-314: DEFAULT - Build locally, upload binary (same as install)
    n!("build_mode", "🏗️  Building locally on keeper, will upload to '{}'", host);
    
    // Build locally
    n!("cargo_build", "🔨 Building rbee-hive locally...");
    let mut cmd = std::process::Command::new("cargo");
    cmd.arg("build")
        .arg("--release")
        .arg("--bin")
        .arg("rbee-hive");

    let output = cmd.output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        n!("build_failed", "❌ Build failed: {}", stderr);
        anyhow::bail!("Build failed");
    }
    
    n!("build_success", "✅ Build successful!");
    
    // Upload binary
    let local_binary = "target/release/rbee-hive";
    let install_dir = DEFAULT_INSTALL_DIR;
    let remote_path = format!("{}/rbee-hive", install_dir);
    
    n!("upload_binary", "📤 Uploading binary to '{}'...", host);
    client
        .upload_file(local_binary, &remote_path)
        .await?;
    
    // Make executable
    client
        .execute(&format!("chmod +x {}", remote_path))
        .await?;
    
    n!("rebuild_complete", "✅ Hive rebuilt on '{}'", host);
    n!("restart_hint", "💡 Restart hive to use the new binary: rbee hive start -a {}", host);
    
    Ok(())
}

/// Rebuild rbee-hive remotely via SSH with on-site build
///
/// TEAM-314: Optional mode (--build-remote) - build on remote host
async fn rebuild_hive_remote_onsite(host: &str, client: &SshClient) -> Result<()> {
    n!("build_mode", "🏗️  Building on-site (requires git)");
    
    // Check if cargo is installed
    n!("check_cargo", "🔍 Checking for cargo...");
    let cargo_check = client
        .execute("source $HOME/.cargo/env 2>/dev/null && which cargo || which cargo")
        .await;
    
    if cargo_check.is_err() {
        n!("cargo_missing", "❌ Cargo not found");
        anyhow::bail!(
            "Cargo not found on remote host.\n\n\
             Install Rust first:\n\
             curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        );
    }
    n!("cargo_found", "✅ Cargo found");
    
    // TEAM-314: PARITY - Clone fresh (same as install)
    // Remove old build dir if exists
    let build_dir = DEFAULT_BUILD_DIR;
    n!("git_clone", "📥 Cloning repo to {} (shallow clone)...", build_dir);
    let _ = client.execute(&format!("rm -rf {}", build_dir)).await;
    
    // Get the git remote URL from the current repo (same as install)
    n!("detect_repo", "🔍 Detecting git repository URL...");
    let git_url = std::process::Command::new("git")
        .args(["config", "--get", "remote.origin.url"])
        .output()?;
    
    let git_url = String::from_utf8_lossy(&git_url.stdout).trim().to_string();
    
    if git_url.is_empty() {
        anyhow::bail!("No git remote URL found. Make sure you're in a git repository.");
    }
    
    n!("repo_url", "📍 Repository URL: {}", git_url);
    
    // Shallow clone (depth=1, no history)
    client
        .execute(&format!(
            "git clone --depth 1 {} {}",
            git_url, build_dir
        ))
        .await?;
    n!("git_clone_complete", "✅ Clone complete");
    
    // Build on remote
    n!("cargo_build", "🔨 Building rbee-hive on '{}' (this may take 5-10 minutes)...", host);
    n!("cargo_build_wait", "⏳ Please wait - cargo output will appear when build completes...");
    
    let build_output = client
        .execute(&format!(
            "source $HOME/.cargo/env && cd {} && cargo build --release --bin rbee-hive 2>&1 | tee rebuild.log",
            build_dir
        ))
        .await?;
    
    // Show last 20 lines of build output
    let lines: Vec<&str> = build_output.lines().collect();
    let last_lines = if lines.len() > 20 {
        &lines[lines.len()-20..]
    } else {
        &lines[..]
    };
    
    n!("cargo_build_output", "📋 Build output (last 20 lines):");
    for line in last_lines {
        println!("{}", line);
    }
    
    n!("cargo_build_complete", "✅ Build complete");
    
    // Install binary
    n!("install_binary", "📦 Installing updated binary...");
    
    let install_dir = crate::DEFAULT_INSTALL_DIR;
    let remote_path = format!("{}/rbee-hive", install_dir);
    
    client
        .execute(&format!(
            "cp {}/target/release/rbee-hive {}",
            build_dir, remote_path
        ))
        .await?;
    
    // TEAM-314: DON'T cleanup build directory - keep for faster rebuilds
    // The directory will be removed and re-cloned on next install/rebuild anyway
    n!("build_dir_kept", "💡 Build directory kept at {} for faster rebuilds", build_dir);
    
    n!("rebuild_complete", "✅ Hive rebuilt on '{}'", host);
    n!("restart_hint", "💡 Restart hive to use the new binary: rbee hive start -a {}", host);
    
    Ok(())
}
