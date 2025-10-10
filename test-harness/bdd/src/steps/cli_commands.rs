// CLI command step definitions
// Created by: TEAM-040
// Modified by: TEAM-043 (added real command execution)

use crate::steps::world::World;
use cucumber::{given, then, when};

#[given(expr = "the following config files exist:")]
pub async fn given_config_files_exist(world: &mut World, step: &cucumber::gherkin::Step) {
    let table = step.table.as_ref().expect("Expected a data table");
    tracing::debug!("Config files exist: {} entries", table.rows.len() - 1);
}

#[given(expr = "config file contains:")]
pub async fn given_config_contains(world: &mut World, step: &cucumber::gherkin::Step) {
    let docstring = step.docstring.as_ref().expect("Expected a docstring");
    tracing::debug!("Config contains: {}", docstring.trim());
}

#[when(expr = "I run {string}")]
pub async fn when_i_run_command_string(world: &mut World, command: String) {
    // TEAM-044: Execute the command for real, not just store it
    tracing::info!("🚀 Executing command: {}", command);

    // Parse command line into parts
    let parts: Vec<&str> = command.split_whitespace().collect();
    if parts.is_empty() {
        panic!("Empty command");
    }

    // Extract binary name and args
    let binary = parts[0];
    let args: Vec<&str> = parts[1..].to_vec();

    // Map command names to actual binary names
    let actual_binary = if binary == "rbee-keeper" { "rbee" } else { binary };

    // Use pre-built binaries
    let workspace_dir = std::env::var("CARGO_MANIFEST_DIR")
        .map(|p| std::path::PathBuf::from(p).parent().unwrap().parent().unwrap().to_path_buf())
        .unwrap_or_else(|_| std::path::PathBuf::from("/home/vince/Projects/llama-orch"));

    let binary_path = workspace_dir.join("target/debug").join(actual_binary);

    // Execute command
    let output = tokio::process::Command::new(&binary_path)
        .args(&args)
        .current_dir(&workspace_dir)
        .output()
        .await
        .expect("Failed to execute command");

    world.last_command = Some(command.clone());
    world.last_exit_code = output.status.code();
    world.last_stdout = String::from_utf8_lossy(&output.stdout).to_string();
    world.last_stderr = String::from_utf8_lossy(&output.stderr).to_string();

    tracing::info!("✅ Command completed with exit code: {:?}", world.last_exit_code);
    if !world.last_stdout.is_empty() {
        tracing::info!("stdout: {}", world.last_stdout);
    }
    if !world.last_stderr.is_empty() {
        tracing::warn!("stderr: {}", world.last_stderr);
    }
}

// TEAM-043: Real command execution with docstring
#[when(expr = "I run:")]
pub async fn when_i_run_command_docstring(world: &mut World, step: &cucumber::gherkin::Step) {
    let docstring = step.docstring.as_ref().expect("Expected a docstring");
    let command_line = docstring.trim();

    tracing::info!("🚀 Executing command: {}", command_line);

    // Parse command line into parts
    let parts: Vec<&str> = command_line.split_whitespace().collect();
    if parts.is_empty() {
        panic!("Empty command");
    }

    // Extract binary name and args
    let binary = parts[0];
    let args: Vec<&str> = parts[1..]
        .iter()
        .filter(|s| !s.starts_with('\\')) // Filter out line continuations
        .copied()
        .collect();

    // TEAM-044: Map command names to actual binary names
    // rbee-keeper -> rbee (the actual binary name)
    let actual_binary = if binary == "rbee-keeper" { "rbee" } else { binary };

    // TEAM-044: Use pre-built binaries instead of cargo run to avoid compilation timeouts
    let workspace_dir = std::env::var("CARGO_MANIFEST_DIR")
        .map(|p| std::path::PathBuf::from(p).parent().unwrap().parent().unwrap().to_path_buf())
        .unwrap_or_else(|_| std::path::PathBuf::from("/home/vince/Projects/llama-orch"));

    let binary_path = workspace_dir.join("target/debug").join(actual_binary);

    // Execute command using pre-built binary
    let output = tokio::process::Command::new(&binary_path)
        .args(&args)
        .current_dir(&workspace_dir)
        .output()
        .await
        .expect("Failed to execute command");

    world.last_command = Some(command_line.to_string());
    world.last_exit_code = output.status.code();
    world.last_stdout = String::from_utf8_lossy(&output.stdout).to_string();
    world.last_stderr = String::from_utf8_lossy(&output.stderr).to_string();

    tracing::info!("✅ Command completed with exit code: {:?}", world.last_exit_code);
    if !world.last_stdout.is_empty() {
        tracing::info!("stdout: {}", world.last_stdout);
    }
    if !world.last_stderr.is_empty() {
        tracing::warn!("stderr: {}", world.last_stderr);
    }
}

#[when(expr = "RBEE_CONFIG={string} is set")]
pub async fn when_rbee_config_set(world: &mut World, path: String) {
    tracing::debug!("RBEE_CONFIG set to: {}", path);
}

#[when(expr = "RBEE_CONFIG is not set")]
pub async fn when_rbee_config_not_set(world: &mut World) {
    tracing::debug!("RBEE_CONFIG is not set");
}

#[when(expr = "{string} exists")]
pub async fn when_file_exists(world: &mut World, path: String) {
    tracing::debug!("File exists: {}", path);
}

#[when(expr = "neither RBEE_CONFIG nor user config exist")]
pub async fn when_neither_config_exists(world: &mut World) {
    tracing::debug!("Neither RBEE_CONFIG nor user config exist");
}

#[when(expr = "rbee-keeper executes remote command on {string}")]
pub async fn when_execute_remote_command(world: &mut World, node: String) {
    tracing::debug!("Executing remote command on: {}", node);
}

#[then(expr = "binaries are installed to {string}")]
pub async fn then_binaries_installed_to(world: &mut World, path: String) {
    tracing::debug!("Binaries should be installed to: {}", path);
}

#[then(expr = "config directory is created at {string}")]
pub async fn then_config_dir_created(world: &mut World, path: String) {
    tracing::debug!("Config directory should be created at: {}", path);
}

#[then(expr = "data directory is created at {string}")]
pub async fn then_data_dir_created(world: &mut World, path: String) {
    tracing::debug!("Data directory should be created at: {}", path);
}

#[then(expr = "default config file is generated at {string}")]
pub async fn then_default_config_generated(world: &mut World, path: String) {
    tracing::debug!("Default config should be generated at: {}", path);
}

#[then(expr = "the following binaries are copied:")]
pub async fn then_binaries_copied(world: &mut World, step: &cucumber::gherkin::Step) {
    let table = step.table.as_ref().expect("Expected a data table");
    tracing::debug!("Binaries should be copied: {} entries", table.rows.len() - 1);
}

#[then(expr = "installation instructions are displayed")]
pub async fn then_installation_instructions(world: &mut World) {
    tracing::debug!("Installation instructions should be displayed");
}

#[then(expr = "sudo permissions are required")]
pub async fn then_sudo_required(world: &mut World) {
    tracing::debug!("Sudo permissions should be required");
}

#[then(expr = "rbee-keeper loads config from {string}")]
pub async fn then_load_config_from(world: &mut World, path: String) {
    tracing::debug!("Should load config from: {}", path);
}

#[then(expr = "the command uses {string} instead of {string}")]
pub async fn then_command_uses_instead(world: &mut World, actual: String, default: String) {
    tracing::debug!("Command should use '{}' instead of '{}'", actual, default);
}

#[then(expr = "git commands use {string} instead of {string}")]
pub async fn then_git_uses_instead(world: &mut World, actual: String, default: String) {
    tracing::debug!("Git commands should use '{}' instead of '{}'", actual, default);
}

#[then(expr = "the command executes the full inference flow")]
pub async fn then_execute_full_flow(world: &mut World) {
    tracing::debug!("Should execute full inference flow");
}

#[then(expr = "tokens are streamed to stdout")]
pub async fn then_tokens_streamed_stdout(world: &mut World) {
    tracing::debug!("Tokens should be streamed to stdout");
}

#[given(expr = "workers are registered on multiple nodes")]
pub async fn given_workers_on_multiple_nodes(world: &mut World) {
    tracing::debug!("Workers registered on multiple nodes");
}

#[given(regex = r#"^a worker with id "(.+)" is running$"#)]
pub async fn given_worker_with_id_running(world: &mut World, worker_id: String) {
    tracing::debug!("Worker {} is running", worker_id);
}

#[then(expr = "the output shows health status of workers on mac")]
pub async fn then_output_shows_health_status(world: &mut World) {
    tracing::debug!("Output should show health status");
}

#[then(expr = "logs from mac are streamed to stdout")]
pub async fn then_logs_streamed(world: &mut World) {
    tracing::debug!("Logs should be streamed to stdout");
}

// TEAM-043: Exit code verification
#[then(expr = "the exit code is {int}")]
pub async fn then_exit_code_is(world: &mut World, expected_code: i32) {
    assert_eq!(
        world.last_exit_code,
        Some(expected_code),
        "Expected exit code {}, got {:?}",
        expected_code,
        world.last_exit_code
    );
    tracing::info!("✅ Exit code is {}", expected_code);
}
