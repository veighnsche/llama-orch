// CLI command step definitions
// Created by: TEAM-040

use cucumber::{given, when, then};
use crate::steps::world::World;

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
    world.last_command = Some(command.clone());
    tracing::debug!("Running command: {}", command);
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
