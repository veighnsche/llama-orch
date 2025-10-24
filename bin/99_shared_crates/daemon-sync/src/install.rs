//! Installation logic for hives and workers via SSH
//!
//! Created by: TEAM-280
//!
//! This module handles SSH-based installation of:
//! - Hive binaries (rbee-hive) on remote hosts
//! - Worker binaries (rbee-worker-*) on remote hosts
//!
//! Key insight: Queen installs BOTH hive AND workers via SSH.
//! This is different from the old architecture where hive installed workers.

use anyhow::{Context, Result};
use observability_narration_core::NarrationFactory;
use queen_rbee_ssh_client::RbeeSSHClient;
use rbee_config::declarative::{HiveConfig, InstallMethod, WorkerConfig};
use std::sync::Arc;

const NARRATE: NarrationFactory = NarrationFactory::new("pkg-inst");

/// Install hive binary on remote host via SSH
///
/// TEAM-280: Queen installs hive binary remotely
///
/// # Steps
/// 1. Connect via SSH
/// 2. Create directory structure
/// 3. Download hive binary
/// 4. Set executable permissions
/// 5. Verify installation
///
/// # Arguments
/// * `hive` - Hive configuration
/// * `job_id` - Job ID for SSE routing
///
/// # Returns
/// * `Ok(())` - Hive installed successfully
/// * `Err` - Installation failed
pub async fn install_hive_binary(hive: &HiveConfig, job_id: &str) -> Result<()> {
    NARRATE
        .action("install_hive")
        .job_id(job_id)
        .context(&hive.alias)
        .human("📦 Installing hive binary on '{}'")
        .emit();

    // Connect via SSH
    let mut client = RbeeSSHClient::connect(&hive.hostname, hive.ssh_port, &hive.ssh_user)
        .await
        .context(format!("Failed to connect to {}", hive.hostname))?;

    // Create directory structure
    // Note: russh exec doesn't use a login shell, so we need absolute paths or $HOME
    let setup_cmd = "mkdir -p ~/.local/bin ~/.local/share/rbee/hives";
    let (stdout, stderr, exit_code) = client.exec(setup_cmd).await?;
    if exit_code != 0 {
        client.close().await?;
        return Err(anyhow::anyhow!(
            "Failed to create directories (exit {}): stdout='{}' stderr='{}'",
            exit_code, stdout, stderr
        ));
    }

    // Determine binary path based on install method
    let binary_path = if let Some(path) = &hive.binary_path {
        // Use custom path
        path.clone()
    } else {
        match &hive.install_method {
            InstallMethod::Git { repo, branch } => {
                install_hive_from_git(&mut client, repo, branch, job_id, &hive.alias).await?
            }
            InstallMethod::Release { repo, tag } => {
                install_hive_from_release(&mut client, repo, tag, job_id, &hive.alias).await?
            }
            InstallMethod::Local { path } => {
                // TEAM-260: Install from local path (copy binary via SCP)
                install_hive_from_local(&mut client, path, job_id, &hive.alias).await?
            }
        }
    };

    // Verify installation
    let verify_cmd = format!("{} --version", binary_path);
    let (stdout, stderr, exit_code) = client.exec(&verify_cmd).await?;

    client.close().await?;

    if exit_code != 0 {
        return Err(anyhow::anyhow!("Hive binary verification failed: {}", stderr));
    }

    NARRATE
        .action("hive_installed")
        .job_id(job_id)
        .context(&hive.alias)
        .context(stdout.trim())
        .human("✅ Hive '{}' installed: {}")
        .emit();

    Ok(())
}

/// Install worker binary on remote host via SSH
///
/// TEAM-280: Queen installs worker binaries remotely (NEW!)
///
/// This is a key architectural change: workers are installed by queen via SSH,
/// not by the hive itself. This simplifies the hive's responsibilities.
///
/// # Steps
/// 1. Connect via SSH
/// 2. Create worker directory
/// 3. Download worker binary
/// 4. Set executable permissions
/// 5. Verify installation
///
/// # Arguments
/// * `hive` - Hive configuration (for SSH connection)
/// * `worker` - Worker configuration
/// * `job_id` - Job ID for SSE routing
///
/// # Returns
/// * `Ok(())` - Worker installed successfully
/// * `Err` - Installation failed
pub async fn install_worker_binary(
    hive: &HiveConfig,
    worker: &WorkerConfig,
    job_id: &str,
) -> Result<()> {
    NARRATE
        .action("install_worker")
        .job_id(job_id)
        .context(&hive.alias)
        .context(&worker.worker_type)
        .human("📦 Installing worker '{}' on hive '{}'")
        .emit();

    // Connect via SSH
    let mut client = RbeeSSHClient::connect(&hive.hostname, hive.ssh_port, &hive.ssh_user)
        .await
        .context(format!("Failed to connect to {}", hive.hostname))?;

    // Create worker directory
    let setup_cmd = "mkdir -p ~/.local/share/rbee/workers";
    let (_, stderr, exit_code) = client.exec(setup_cmd).await?;
    if exit_code != 0 {
        client.close().await?;
        return Err(anyhow::anyhow!("Failed to create worker directory: {}", stderr));
    }

    // Determine binary path based on install method
    let binary_path = if let Some(path) = &worker.binary_path {
        // Use custom path
        path.clone()
    } else {
        match &worker.install_method {
            InstallMethod::Git { repo, branch } => {
                install_worker_from_git(
                    &mut client,
                    repo,
                    branch,
                    &worker.worker_type,
                    &worker.features,
                    job_id,
                    &hive.alias,
                )
                .await?
            }
            InstallMethod::Release { repo, tag } => {
                install_worker_from_release(
                    &mut client,
                    repo,
                    tag,
                    &worker.worker_type,
                    job_id,
                    &hive.alias,
                )
                .await?
            }
            InstallMethod::Local { path } => path.clone(),
        }
    };

    // Verify installation
    let verify_cmd = format!("{} --version", binary_path);
    let (stdout, stderr, exit_code) = client.exec(&verify_cmd).await?;

    client.close().await?;

    if exit_code != 0 {
        return Err(anyhow::anyhow!("Worker binary verification failed: {}", stderr));
    }

    NARRATE
        .action("worker_installed")
        .job_id(job_id)
        .context(&worker.worker_type)
        .context(&hive.alias)
        .context(stdout.trim())
        .human("✅ Worker '{}' installed on '{}': {}")
        .emit();

    Ok(())
}

/// Install all components for a hive (hive + workers) concurrently
///
/// TEAM-280: Concurrent installation for speed
///
/// # Steps
/// 1. Install hive binary
/// 2. Install all workers concurrently (tokio::spawn)
/// 3. Collect results
///
/// # Arguments
/// * `hive` - Hive configuration
/// * `job_id` - Job ID for SSE routing
///
/// # Returns
/// * `Ok(())` - All components installed successfully
/// * `Err` - Installation failed
pub async fn install_all(hive: Arc<HiveConfig>, job_id: String) -> Result<()> {
    NARRATE
        .action("install_all")
        .job_id(&job_id)
        .context(&hive.alias)
        .human("🚀 Installing all components for hive '{}'")
        .emit();

    // Step 1: Install hive binary first
    install_hive_binary(&hive, &job_id).await?;

    // Step 2: Install all workers concurrently
    if !hive.workers.is_empty() {
        NARRATE
            .action("install_workers")
            .job_id(&job_id)
            .context(&hive.alias)
            .context(hive.workers.len().to_string())
            .human("📦 Installing {} workers for '{}'")
            .emit();

        let mut tasks = Vec::new();
        for worker in &hive.workers {
            let hive_clone = Arc::clone(&hive);
            let worker_clone = worker.clone();
            let job_id_clone = job_id.clone();

            let task = tokio::spawn(async move {
                install_worker_binary(&hive_clone, &worker_clone, &job_id_clone).await
            });
            tasks.push(task);
        }

        // Wait for all workers to install
        let results = futures::future::join_all(tasks).await;

        // Check for failures
        for (idx, result) in results.into_iter().enumerate() {
            match result {
                Ok(Ok(())) => {
                    // Worker installed successfully
                }
                Ok(Err(e)) => {
                    return Err(anyhow::anyhow!(
                        "Worker {} installation failed: {}",
                        hive.workers[idx].worker_type,
                        e
                    ));
                }
                Err(e) => {
                    return Err(anyhow::anyhow!("Worker task panicked: {}", e));
                }
            }
        }
    }

    NARRATE
        .action("install_complete")
        .job_id(&job_id)
        .context(&hive.alias)
        .human("✅ All components installed for hive '{}'")
        .emit();

    Ok(())
}

/// Install hive binary from git repository
///
/// Clones repo (shallow), builds rbee-hive, and installs to ~/.local/bin
///
/// # Arguments
/// * `client` - SSH client connection
/// * `repo` - Git repository URL
/// * `branch` - Git branch/tag/commit
/// * `job_id` - Job ID for SSE routing
/// * `alias` - Hive alias for logging
async fn install_hive_from_git(
    client: &mut RbeeSSHClient,
    repo: &str,
    branch: &str,
    job_id: &str,
    alias: &str,
) -> Result<String> {
    NARRATE
        .action("git_clone_hive")
        .job_id(job_id)
        .context(alias)
        .context(repo)
        .human("📥 Cloning repository for hive '{}': {}")
        .emit();

    // Clone repository (shallow, no history for speed)
    let clone_dir = "~/.local/share/rbee/build";
    let clone_cmd = format!(
        "rm -rf {} && mkdir -p {} && git clone --depth 1 --branch {} {} {}",
        clone_dir, clone_dir, branch, repo, clone_dir
    );

    // ============================================================
    // TEAM-260: INVESTIGATION - SSH exec fails silently
    // ============================================================
    // SUSPICION:
    // - SSH exec call hangs or fails immediately
    // - Error is propagated with ? but never visible to user
    // - Spawned task catches error but narration doesn't reach SSE stream
    //
    // INVESTIGATION:
    // - Manual SSH command works perfectly
    // - Product code fails at client.exec().await?
    // - Narration shows git_clone_exec but NOT git_clone_result
    // - This means the await? never returns (either hangs or errors)
    // 
    // DEBUGGING:
    // - Adding synchronous eprintln! to capture actual error
    // - This will show in queen-rbee stderr even if narration is lost
    // ============================================================
    
    NARRATE
        .action("git_clone_exec")
        .job_id(job_id)
        .context(alias)
        .context(&clone_cmd)
        .human("🔧 Executing clone command for '{}': {}")
        .emit();

    eprintln!("[TEAM-260 DEBUG] About to execute SSH command for hive '{}'", alias);
    eprintln!("[TEAM-260 DEBUG] Command: {}", clone_cmd);
    
    let exec_result = client.exec(&clone_cmd).await;
    
    eprintln!("[TEAM-260 DEBUG] SSH exec returned for hive '{}'", alias);
    
    let (stdout, stderr, exit_code) = match exec_result {
        Ok(result) => {
            eprintln!("[TEAM-260 DEBUG] SSH exec SUCCESS: exit={}", result.2);
            result
        }
        Err(e) => {
            eprintln!("[TEAM-260 DEBUG] SSH exec FAILED: {}", e);
            eprintln!("[TEAM-260 DEBUG] Error details: {:?}", e);
            
            // Emit error narration
            NARRATE
                .action("ssh_exec_error")
                .job_id(job_id)
                .context(alias)
                .context(&format!("{}", e))
                .human("❌ SSH exec failed for '{}': {}")
                .error_kind("ssh_exec_failed")
                .emit();
            
            return Err(e);
        }
    };
    
    NARRATE
        .action("git_clone_result")
        .job_id(job_id)
        .context(alias)
        .context(&format!("exit={}", exit_code))
        .human("📊 Clone command completed for '{}': exit={}")
        .emit();
    if exit_code != 0 {
        // TEAM-260: Emit detailed error narration
        NARRATE
            .action("git_clone_failed")
            .job_id(job_id)
            .context(alias)
            .context(&format!("exit_code={}", exit_code))
            .context(&format!("stdout={}", stdout))
            .context(&format!("stderr={}", stderr))
            .human("❌ Git clone failed for '{}': exit={}, stdout={}, stderr={}")
            .error_kind("git_clone_failed")
            .emit();
        
        return Err(anyhow::anyhow!(
            "Failed to clone repository (exit {}): stdout='{}' stderr='{}'",
            exit_code, stdout, stderr
        ));
    }

    NARRATE
        .action("git_clone_complete")
        .job_id(job_id)
        .context(alias)
        .human("✅ Repository cloned")
        .emit();

    NARRATE
        .action("build_hive")
        .job_id(job_id)
        .context(alias)
        .human("🔨 Building hive binary for '{}'")
        .emit();

    // Build rbee-hive binary
    // TEAM-260: Redirect output to log file to prevent SSH buffer issues during long builds
    let build_log = format!("{}/.build.log", clone_dir);
    let build_cmd = format!(
        "cd {} && cargo build --release --bin rbee-hive > {} 2>&1; echo $?",
        clone_dir, build_log
    );

    let (stdout, stderr, exit_code) = client.exec(&build_cmd).await?;
    
    // The command should always succeed (we capture exit code with echo $?)
    // Parse the actual build exit code from stdout
    let build_exit_code: i32 = stdout.trim().parse().unwrap_or(255);
    
    if build_exit_code != 0 {
        // Read the build log to get error details
        let log_cmd = format!("tail -100 {}", build_log);
        let (log_output, _, _) = client.exec(&log_cmd).await.unwrap_or_default();
        
        return Err(anyhow::anyhow!(
            "Failed to build hive binary (exit {}): {}",
            build_exit_code, log_output
        ));
    }

    NARRATE
        .action("build_complete")
        .job_id(job_id)
        .context(alias)
        .human("✅ Binary built successfully")
        .emit();

    // Copy binary to ~/.local/bin
    let install_cmd = format!(
        "cp {}/target/release/rbee-hive ~/.local/bin/rbee-hive && chmod +x ~/.local/bin/rbee-hive",
        clone_dir
    );

    let (_, stderr, exit_code) = client.exec(&install_cmd).await?;
    if exit_code != 0 {
        return Err(anyhow::anyhow!("Failed to install hive binary: {}", stderr));
    }

    NARRATE
        .action("hive_built")
        .job_id(job_id)
        .context(alias)
        .human("✅ Hive binary built and installed for '{}'")
        .emit();

    Ok("~/.local/bin/rbee-hive".to_string())
}

/// Install hive binary from local path (copy via SCP)
///
/// TEAM-260: Added for robust testing and local development
///
/// # Arguments
/// * `client` - SSH client connection
/// * `local_path` - Path to local binary on HOST
/// * `job_id` - Job ID for SSE routing
/// * `alias` - Hive alias for logging
async fn install_hive_from_local(
    client: &mut RbeeSSHClient,
    local_path: &str,
    job_id: &str,
    alias: &str,
) -> Result<String> {
    NARRATE
        .action("local_install_hive")
        .job_id(job_id)
        .context(alias)
        .context(local_path)
        .human("📦 Installing hive from local path '{}': {}")
        .emit();

    // TEAM-260: Resolve relative paths to absolute paths
    // SCP requires absolute paths or paths relative to current working directory
    let absolute_path = if std::path::Path::new(local_path).is_absolute() {
        local_path.to_string()
    } else {
        std::env::current_dir()
            .context("Failed to get current directory")?
            .join(local_path)
            .to_str()
            .context("Path contains invalid UTF-8")?
            .to_string()
    };
    
    // Verify local file exists
    if !std::path::Path::new(&absolute_path).exists() {
        return Err(anyhow::anyhow!(
            "Local binary not found: {} (resolved from: {})",
            absolute_path, local_path
        ));
    }

    NARRATE
        .action("local_binary_found")
        .job_id(job_id)
        .context(alias)
        .context(&absolute_path)
        .human("✅ Local binary found for '{}': {}")
        .emit();

    // Create directory
    let mkdir_cmd = "mkdir -p ~/.local/bin";
    let (_, stderr, exit_code) = client.exec(mkdir_cmd).await?;
    if exit_code != 0 {
        return Err(anyhow::anyhow!("Failed to create directory: {}", stderr));
    }

    // Copy binary via SCP
    let remote_path = "~/.local/bin/rbee-hive";
    client.copy_file(&absolute_path, remote_path).await?;

    // Make executable
    let chmod_cmd = "chmod +x ~/.local/bin/rbee-hive";
    let (_, stderr, exit_code) = client.exec(chmod_cmd).await?;
    if exit_code != 0 {
        return Err(anyhow::anyhow!("Failed to chmod: {}", stderr));
    }

    NARRATE
        .action("local_install_complete")
        .job_id(job_id)
        .context(alias)
        .human("✅ Hive binary installed from local path for '{}'")
        .emit();

    Ok("~/.local/bin/rbee-hive".to_string())
}

/// Install worker binary from git repository
///
/// Clones repo (shallow), builds worker with features, and installs to ~/.local/share/rbee/workers
///
/// # Arguments
/// * `client` - SSH client connection
/// * `repo` - Git repository URL
/// * `branch` - Git branch/tag/commit
/// * `worker_type` - Worker type (e.g., "vllm", "llama-cpp")
/// * `features` - Cargo feature flags (e.g., ["cuda"], ["metal"], ["cpu"])
/// * `job_id` - Job ID for SSE routing
/// * `alias` - Hive alias for logging
async fn install_worker_from_git(
    client: &mut RbeeSSHClient,
    repo: &str,
    branch: &str,
    worker_type: &str,
    features: &[String],
    job_id: &str,
    alias: &str,
) -> Result<String> {
    NARRATE
        .action("git_clone_worker")
        .job_id(job_id)
        .context(worker_type)
        .context(alias)
        .context(repo)
        .human("📥 Cloning repository for worker '{}' on '{}': {}")
        .emit();

    // Clone repository (shallow, no history for speed)
    let clone_dir = "~/.local/share/rbee/build";
    let clone_cmd = format!(
        "rm -rf {} && mkdir -p {} && git clone --depth 1 --branch {} {} {}",
        clone_dir, clone_dir, branch, repo, clone_dir
    );

    let (_, stderr, exit_code) = client.exec(&clone_cmd).await?;
    if exit_code != 0 {
        return Err(anyhow::anyhow!("Failed to clone repository: {}", stderr));
    }

    NARRATE
        .action("build_worker")
        .job_id(job_id)
        .context(worker_type)
        .context(alias)
        .human("🔨 Building worker '{}' for '{}' with features: {:?}")
        .context(format!("{:?}", features))
        .emit();

    // Build worker binary with features
    let features_flag = if features.is_empty() {
        String::new()
    } else {
        format!("--features {}", features.join(","))
    };

    let build_cmd = format!(
        "cd {} && cargo build --release --bin llm-worker-rbee {}",
        clone_dir, features_flag
    );

    let (_, stderr, exit_code) = client.exec(&build_cmd).await?;
    if exit_code != 0 {
        return Err(anyhow::anyhow!("Failed to build worker binary: {}", stderr));
    }

    // Copy binary to ~/.local/share/rbee/workers with type-specific name
    let binary_name = format!("rbee-worker-{}", worker_type);
    let target_path = format!("~/.local/share/rbee/workers/{}", binary_name);
    let install_cmd = format!(
        "cp {}/target/release/llm-worker-rbee {} && chmod +x {}",
        clone_dir, target_path, target_path
    );

    let (_, stderr, exit_code) = client.exec(&install_cmd).await?;
    if exit_code != 0 {
        return Err(anyhow::anyhow!("Failed to install worker binary: {}", stderr));
    }

    NARRATE
        .action("worker_built")
        .job_id(job_id)
        .context(worker_type)
        .context(alias)
        .human("✅ Worker '{}' built and installed for '{}'")
        .emit();

    Ok(target_path)
}

/// Install hive binary from GitHub release
///
/// Downloads pre-built binary from GitHub releases
async fn install_hive_from_release(
    client: &mut RbeeSSHClient,
    repo: &str,
    tag: &str,
    job_id: &str,
    alias: &str,
) -> Result<String> {
    NARRATE
        .action("download_hive_release")
        .job_id(job_id)
        .context(alias)
        .context(repo)
        .context(tag)
        .human("⬇️  Downloading hive release for '{}': {} @ {}")
        .emit();

    let download_url = format!("https://github.com/{}/releases/download/{}/rbee-hive", repo, tag);

    let download_cmd = format!(
        "curl -L {} -o ~/.local/bin/rbee-hive && chmod +x ~/.local/bin/rbee-hive",
        download_url
    );

    let (_, stderr, exit_code) = client.exec(&download_cmd).await?;
    if exit_code != 0 {
        return Err(anyhow::anyhow!("Failed to download hive release: {}", stderr));
    }

    Ok("~/.local/bin/rbee-hive".to_string())
}

/// Install worker binary from GitHub release
///
/// Downloads pre-built binary from GitHub releases
async fn install_worker_from_release(
    client: &mut RbeeSSHClient,
    repo: &str,
    tag: &str,
    worker_type: &str,
    job_id: &str,
    alias: &str,
) -> Result<String> {
    NARRATE
        .action("download_worker_release")
        .job_id(job_id)
        .context(worker_type)
        .context(alias)
        .context(repo)
        .context(tag)
        .human("⬇️  Downloading worker '{}' release for '{}': {} @ {}")
        .emit();

    let binary_name = format!("rbee-worker-{}", worker_type);
    let download_url =
        format!("https://github.com/{}/releases/download/{}/{}", repo, tag, binary_name);

    let target_path = format!("~/.local/share/rbee/workers/{}", binary_name);
    let download_cmd =
        format!("curl -L {} -o {} && chmod +x {}", download_url, target_path, target_path);

    let (_, stderr, exit_code) = client.exec(&download_cmd).await?;
    if exit_code != 0 {
        return Err(anyhow::anyhow!("Failed to download worker release: {}", stderr));
    }

    Ok(target_path)
}
