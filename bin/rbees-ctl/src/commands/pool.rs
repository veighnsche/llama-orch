//! Pool management commands (via SSH)
//!
//! Created by: TEAM-022

use crate::cli::{GitAction, ModelsAction, PoolAction, WorkerAction};
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
    let command = match action {
        ModelsAction::Download { model } => {
            format!(
                "cd ~/Projects/llama-orch && ./target/release/rbees-pool models download {}",
                model
            )
        }
        ModelsAction::List => {
            "cd ~/Projects/llama-orch && ./target/release/rbees-pool models list".to_string()
        }
        ModelsAction::Catalog => {
            "cd ~/Projects/llama-orch && ./target/release/rbees-pool models catalog".to_string()
        }
        ModelsAction::Register { id, name, repo, architecture } => {
            format!(
                "cd ~/Projects/llama-orch && ./target/release/rbees-pool models register {} --name '{}' --repo '{}' --architecture {}",
                id, name, repo, architecture
            )
        }
    };

    // ssh2 now handles the "→ SSH to host" message with spinner
    ssh::execute_remote_command_streaming(host, &command)?;
    Ok(())
}

fn handle_worker(action: WorkerAction, host: &str) -> Result<()> {
    let command = match action {
        WorkerAction::Spawn { backend, model, gpu } => {
            format!(
                "cd ~/Projects/llama-orch && ./target/release/rbees-pool worker spawn {} --model {} --gpu {}",
                backend, model, gpu
            )
        }
        WorkerAction::List => {
            "cd ~/Projects/llama-orch && ./target/release/rbees-pool worker list".to_string()
        }
        WorkerAction::Stop { worker_id } => {
            format!(
                "cd ~/Projects/llama-orch && ./target/release/rbees-pool worker stop {}",
                worker_id
            )
        }
    };

    ssh::execute_remote_command_streaming(host, &command)?;
    Ok(())
}

fn handle_git(action: GitAction, host: &str) -> Result<()> {
    let command = match action {
        GitAction::Pull => "cd ~/Projects/llama-orch && git pull".to_string(),
        GitAction::Status => "cd ~/Projects/llama-orch && git status".to_string(),
        GitAction::Build => {
            "cd ~/Projects/llama-orch && cargo build --release -p rbees-pool".to_string()
        }
    };

    ssh::execute_remote_command_streaming(host, &command)?;
    Ok(())
}

fn handle_status(host: &str) -> Result<()> {
    let command = "cd ~/Projects/llama-orch && ./target/release/rbees-pool status";

    ssh::execute_remote_command_streaming(host, command)?;
    Ok(())
}
