//! Pool management commands (via SSH)
//!
//! Created by: TEAM-022
//! Modified by: TEAM-036 (removed hardcoded paths, use binaries from PATH)

use crate::cli::{GitAction, ModelsAction, PoolAction, WorkerAction};
use crate::config::Config;
use crate::ssh;
use anyhow::Result;

pub fn handle(action: PoolAction) -> Result<()> {
    match action {
        PoolAction::Models { action, host } => handle_models(action, &host),
        PoolAction::Worker { action, host } => handle_worker(action, &host),
        PoolAction::Git { action, host } => handle_git(action, &host),
        PoolAction::Status { host } => handle_status(&host),
    }
}

fn handle_models(action: ModelsAction, host: &str) -> Result<()> {
    // TEAM-036: Use binary from PATH, with optional config override
    let binary = get_remote_binary_path();

    let command = match action {
        ModelsAction::Download { model } => {
            format!("{} models download {}", binary, model)
        }
        ModelsAction::List => {
            format!("{} models list", binary)
        }
        ModelsAction::Catalog => {
            format!("{} models catalog", binary)
        }
        ModelsAction::Register { id, name, repo, architecture } => {
            format!(
                "{} models register {} --name '{}' --repo '{}' --architecture {}",
                binary, id, name, repo, architecture
            )
        }
    };

    // ssh2 now handles the "→ SSH to host" message with spinner
    ssh::execute_remote_command_streaming(host, &command)?;
    Ok(())
}

fn handle_worker(action: WorkerAction, host: &str) -> Result<()> {
    // TEAM-036: Use binary from PATH, with optional config override
    let binary = get_remote_binary_path();

    let command = match action {
        WorkerAction::Spawn { backend, model, gpu } => {
            format!("{} worker spawn {} --model {} --gpu {}", binary, backend, model, gpu)
        }
        WorkerAction::List => {
            format!("{} worker list", binary)
        }
        WorkerAction::Stop { worker_id } => {
            format!("{} worker stop {}", binary, worker_id)
        }
    };

    ssh::execute_remote_command_streaming(host, &command)?;
    Ok(())
}

fn handle_git(action: GitAction, host: &str) -> Result<()> {
    // TEAM-036: Use configurable repo directory, default to ~/llama-orch
    let repo_dir = get_remote_repo_dir();
    let binary = get_remote_binary_path();

    let command = match action {
        GitAction::Pull => format!("cd {} && git pull", repo_dir),
        GitAction::Status => format!("cd {} && git status", repo_dir),
        GitAction::Build => {
            format!("cd {} && cargo build --release -p rbee-hive", repo_dir)
        }
    };

    ssh::execute_remote_command_streaming(host, &command)?;
    Ok(())
}

fn handle_status(host: &str) -> Result<()> {
    // TEAM-036: Use binary from PATH, with optional config override
    let binary = get_remote_binary_path();
    let command = format!("{} status", binary);

    ssh::execute_remote_command_streaming(host, &command)?;
    Ok(())
}

/// Get the remote binary path from config, or default to PATH
///
/// TEAM-036: Allows custom binary paths via config, defaults to rbee-hive in PATH
fn get_remote_binary_path() -> String {
    Config::load()
        .ok()
        .and_then(|config| config.remote)
        .and_then(|remote| remote.binary_path)
        .unwrap_or_else(|| "rbee-hive".to_string())
}

/// Get the remote repository directory from config, or default to ~/llama-orch
///
/// TEAM-036: Allows custom repo paths via config, defaults to ~/llama-orch
fn get_remote_repo_dir() -> String {
    Config::load()
        .ok()
        .and_then(|config| config.remote)
        .and_then(|remote| remote.git_repo_dir)
        .and_then(|path| path.to_str().map(String::from))
        .unwrap_or_else(|| "~/llama-orch".to_string())
}
