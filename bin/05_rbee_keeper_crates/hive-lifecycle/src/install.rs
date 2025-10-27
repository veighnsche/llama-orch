//! Install rbee-hive on local or remote host
//!
//! TEAM-290: Local or remote hive installation

use anyhow::{Context, Result};
use daemon_lifecycle::{install_daemon, InstallConfig};
use observability_narration_core::n;
use ssh_config::SshClient; // TEAM-314: Use shared SSH client
use std::path::Path;

use crate::{DEFAULT_INSTALL_DIR, DEFAULT_BUILD_DIR};

// TEAM-314: All narration migrated to n!() macro

/// Install rbee-hive on local or remote host
///
/// TEAM-290: Supports both local and remote installation
/// TEAM-314: Default = build locally + upload, Optional = build on-site
///
/// # Arguments
/// * `host` - Host to install on ("localhost" for local, SSH alias for remote)
/// * `binary_path` - Optional path to rbee-hive binary (auto-detects if None)
/// * `install_dir` - Installation directory (default: ~/.local/bin)
/// * `build_remote` - If true, build on remote host (requires git). Default: build locally + upload
///
/// # Example
/// ```rust,ignore
/// // Install locally
/// install_hive("localhost", None, None, false).await?;
///
/// // Install remotely (default: build locally, upload)
/// install_hive("workstation", None, None, false).await?;
///
/// // Install remotely (optional: build on-site)
/// install_hive("workstation", None, None, true).await?;
/// ```
pub async fn install_hive(
    host: &str,
    binary_path: Option<String>,
    install_dir: Option<String>,
    build_remote: bool,
) -> Result<()> {
    n!("install_hive_start", "📦 Installing rbee-hive on '{}'", host);

    // Check if localhost (direct install) or remote (SSH install)
    if host == "localhost" || host == "127.0.0.1" {
        install_hive_local(binary_path, install_dir).await
    } else {
        install_hive_remote(host, binary_path, install_dir, build_remote).await
    }
}

/// Install rbee-hive locally (no SSH)
async fn install_hive_local(
    binary_path: Option<String>,
    install_dir: Option<String>,
) -> Result<()> {
    n!("install_hive_local", "📦 Installing rbee-hive locally...");

    // Use daemon-lifecycle to find/validate binary
    let config = InstallConfig {
        binary_name: "rbee-hive".to_string(),
        binary_path,
        target_path: None,
        job_id: None,
    };

    let install_result = install_daemon(config).await?;
    let source_path = std::path::PathBuf::from(&install_result.binary_path);

    // Determine install location
    let install_dir = if let Some(dir) = install_dir {
        std::path::PathBuf::from(dir)
    } else {
        // Default: ~/.local/bin
        let home = std::env::var("HOME")?;
        std::path::PathBuf::from(format!("{}/.local/bin", home))
    };
    let install_path = install_dir.join("rbee-hive");

    // Create install directory if needed
    std::fs::create_dir_all(&install_dir)?;

    // Copy binary
    std::fs::copy(&source_path, &install_path)
        .context("Failed to copy hive binary")?;

    // Make executable (Unix only)
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&install_path)?.permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&install_path, perms)?;
    }

    // Verify installation
    let output = std::process::Command::new(&install_path)
        .arg("--version")
        .output()
        .context("Failed to verify hive installation")?;

    let version = String::from_utf8_lossy(&output.stdout);

    n!("install_hive_complete", "✅ Hive installed at '{}' (version: {})", 
        install_path.display(), version.trim());

    Ok(())
}

/// Install rbee-hive remotely via SSH
///
/// TEAM-314: DEFAULT = build locally + upload (fast, no git needed)
///           OPTIONAL = build on-site (slower, requires git)
/// 
/// Steps (default):
/// 1. Build locally on keeper
/// 2. Upload binary via SSH
/// 3. Install on remote
///
/// Steps (--build-remote):
/// 1. Git clone on remote
/// 2. Build on remote
/// 3. Install on remote
async fn install_hive_remote(
    host: &str,
    binary_path: Option<String>,
    install_dir: Option<String>,
    build_remote: bool,
) -> Result<()> {
    n!("install_hive_remote", "📦 Installing rbee-hive on '{}' via SSH...", host);

    // Connect via SSH
    let client = SshClient::connect(host).await?;

    // TEAM-314: Use constant default install dir
    let install_dir = install_dir.unwrap_or_else(|| DEFAULT_INSTALL_DIR.to_string());

    // TEAM-314: If build_remote is true, use on-site build (old behavior)
    if build_remote {
        return install_hive_remote_onsite(host, &client, &install_dir).await;
    }

    // TEAM-314: DEFAULT - Build locally, upload binary
    n!("build_mode", "🏗️  Building locally on keeper, will upload to '{}'", host);
    
    // Determine binary path
    let local_binary = if let Some(path) = binary_path {
        // Use provided path
        if !Path::new(&path).exists() {
            anyhow::bail!("Hive binary not found: {}", path);
        }
        path
    } else {
        // Auto-detect: try target/debug first, then target/release
        n!("hive_binary_detect", "🔍 Auto-detecting hive binary...");
        
        let debug_path = "target/debug/rbee-hive";
        let release_path = "target/release/rbee-hive";
        
        if Path::new(debug_path).exists() {
            n!("hive_binary_found", "✅ Found debug binary: {}", debug_path);
            debug_path.to_string()
        } else if Path::new(release_path).exists() {
            n!("hive_binary_found", "✅ Found release binary: {}", release_path);
            release_path.to_string()
        } else {
            n!("hive_binary_missing", "❌ No hive binary found in target/debug or target/release");
            anyhow::bail!(
                "Hive binary not found. Build it first:\n\
                 cargo build --bin rbee-hive\n\
                 Or use --build-remote to build on '{}'",
                host
            );
        }
    };

    // Create install directory
    n!("create_install_dir", "📁 Creating install directory on '{}'...", host);
    client
        .execute(&format!("mkdir -p {}", install_dir))
        .await
        .context("Failed to create install directory")?;

    // Upload binary
    let remote_path = format!("{}/rbee-hive", install_dir);
    n!("upload_binary", "📤 Uploading binary to '{}'...", host);
    client
        .upload_file(&local_binary, &remote_path)
        .await
        .context("Failed to upload hive binary")?;

    // Make executable
    client
        .execute(&format!("chmod +x {}", remote_path))
        .await
        .context("Failed to make hive executable")?;

    // Verify installation
    let version = client
        .execute(&format!("{} --version", remote_path))
        .await
        .context("Failed to verify hive installation")?;

    n!("install_hive_complete", "✅ Hive installed on '{}' (version: {})", 
        host, version.trim());
    n!("install_path", "📍 Binary location: {}", remote_path);
    n!("install_info", "💡 Make sure {} is in PATH on '{}'", install_dir, host);

    Ok(())
}

/// Install rbee-hive remotely via SSH with on-site build
///
/// TEAM-314: Optional mode (--build-remote) - build on remote host
async fn install_hive_remote_onsite(
    host: &str,
    client: &SshClient,
    install_dir: &str,
) -> Result<()> {
    n!("build_mode", "🏗️  Building on-site (hive has the power, keeper might not)");

    // TEAM-314: Build on-site (correct architecture)
    n!("hive_build_onsite", "🏗️  Building on-site (hive has the power, keeper might not)");

    // Step 1: Check if git is installed
    n!("check_git", "🔍 Checking for git...");
    client
        .execute("which git")
        .await
        .context("Git not found on remote host. Install git first: sudo apt install git")?;
    n!("git_found", "✅ Git found");

    // Step 2: Check if cargo is installed
    n!("check_cargo", "🔍 Checking for cargo...");
    
    // TEAM-314: Source cargo env first (SSH doesn't load .bashrc for non-interactive sessions)
    let cargo_check = client
        .execute("source $HOME/.cargo/env 2>/dev/null && which cargo || which cargo")
        .await;
    
    if cargo_check.is_err() {
        n!("cargo_missing", "❌ Cargo not found");
        anyhow::bail!(
            "Cargo not found on remote host.\n\n\
             Install Rust first:\n\
             curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh\n\n\
             Then either:\n\
             1. Logout and login again (to reload PATH)\n\
             2. Or run: source $HOME/.cargo/env"
        );
    }
    n!("cargo_found", "✅ Cargo found");

    // Step 3: Clone repo (shallow, no history)
    // TEAM-314: Use constant for build directory
    let build_dir = DEFAULT_BUILD_DIR;
    n!("git_clone", "📥 Cloning repo to {} (shallow clone)...", build_dir);
    
    // Remove old build dir if exists
    let _ = client.execute(&format!("rm -rf {}", build_dir)).await;
    
    // TEAM-314: Use SSH URL (user has SSH keys set up with GitHub)
    // Get the git remote URL from the current repo
    n!("detect_repo", "🔍 Detecting git repository URL...");
    let git_url = std::process::Command::new("git")
        .args(["config", "--get", "remote.origin.url"])
        .output()
        .context("Failed to get git remote URL")?;
    
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
        .await
        .context("Failed to clone repository")?;
    n!("git_clone_complete", "✅ Clone complete");

    // Step 4: Build on remote
    n!("cargo_build", "🔨 Building rbee-hive on '{}' (this may take 5-10 minutes)...", host);
    n!("cargo_build_wait", "⏳ Please wait - cargo output will appear when build completes...");
    n!("cargo_build_info", "💡 Tip: SSH into '{}' and run 'tail -f /tmp/llama-orch-build/build.log' to watch progress", host);
    
    // TEAM-314: Source cargo env before building (SSH doesn't load .bashrc)
    // Note: SSH execute() waits for completion, so cargo output appears all at once at the end
    // TODO: Future enhancement - stream output in real-time using SSH session
    let build_output = client
        .execute(&format!(
            "source $HOME/.cargo/env && cd {} && cargo build --release --bin rbee-hive 2>&1 | tee build.log",
            build_dir
        ))
        .await
        .context("Failed to build rbee-hive on remote host")?;
    
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

    // Step 5: Install binary
    n!("install_binary", "📦 Installing binary to {}...", install_dir);
    
    // Create install directory
    client
        .execute(&format!("mkdir -p {}", install_dir))
        .await
        .context("Failed to create install directory")?;

    // Copy binary
    let remote_path = format!("{}/rbee-hive", install_dir);
    client
        .execute(&format!(
            "cp {}/target/release/rbee-hive {}",
            build_dir, remote_path
        ))
        .await
        .context("Failed to copy binary")?;

    // Make executable
    client
        .execute(&format!("chmod +x {}", remote_path))
        .await
        .context("Failed to make hive executable")?;

    // TEAM-314: DON'T cleanup build directory - keep for faster rebuilds
    // The directory will be removed and re-cloned on next install/rebuild anyway
    n!("build_dir_kept", "💡 Build directory kept at {} for faster rebuilds", build_dir);

    // Step 7: Verify installation
    let version = client
        .execute(&format!("{} --version", remote_path))
        .await
        .context("Failed to verify hive installation")?;

    n!("install_hive_complete", "✅ Hive installed on '{}' (version: {})", 
        host, version.trim());
    n!("install_path", "📍 Binary location: {}", remote_path);
    n!("install_info", "💡 Make sure {} is in PATH on '{}'", install_dir, host);

    Ok(())
}
