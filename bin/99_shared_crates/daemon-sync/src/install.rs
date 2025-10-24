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
    let setup_cmd = "mkdir -p ~/.local/bin ~/.local/share/rbee/hives";
    let (_, stderr, exit_code) = client.exec(setup_cmd).await?;
    if exit_code != 0 {
        client.close().await?;
        return Err(anyhow::anyhow!("Failed to create directories: {}", stderr));
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
            InstallMethod::Local { path } => path.clone(),
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

    let (_, stderr, exit_code) = client.exec(&clone_cmd).await?;
    if exit_code != 0 {
        return Err(anyhow::anyhow!("Failed to clone repository: {}", stderr));
    }

    NARRATE
        .action("build_hive")
        .job_id(job_id)
        .context(alias)
        .human("🔨 Building hive binary for '{}'")
        .emit();

    // Build rbee-hive binary
    let build_cmd = format!("cd {} && cargo build --release --bin rbee-hive", clone_dir);

    let (_, stderr, exit_code) = client.exec(&build_cmd).await?;
    if exit_code != 0 {
        return Err(anyhow::anyhow!("Failed to build hive binary: {}", stderr));
    }

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
