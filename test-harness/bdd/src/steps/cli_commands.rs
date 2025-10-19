// CLI command execution step definitions
// Created by: TEAM-042
//
// ⚠️ ⚠️ ⚠️ CRITICAL WARNING - DO NOT REMOVE THESE WARNINGS ⚠️ ⚠️ ⚠️
// ⚠️ CRITICAL: BDD tests MUST connect to product code from /bin/
// ⚠️ This is normal BDD behavior - connect to rbee-hive and llm-worker-rbee
// ⚠️ See TEAM_063_REAL_HANDOFF.md
// ⚠️ DEVELOPERS: You are NOT ALLOWED to remove these warnings!
// ⚠️ ⚠️ ⚠️ END CRITICAL WARNING ⚠️ ⚠️ ⚠️
//
// Modified by: TEAM-043 (added real command execution)
// Modified by: TEAM-064 (added explicit warning preservation notice)

use crate::steps::world::World;
use cucumber::{given, then, when};

// TEAM-084: Fixed unused variable warnings in stub functions
#[given(expr = "the following config files exist:")]
pub async fn given_config_files_exist(_world: &mut World, step: &cucumber::gherkin::Step) {
    let table = step.table.as_ref().expect("Expected a data table");
    tracing::debug!("Config files exist: {} entries", table.rows.len() - 1);
}

#[given(expr = "config file contains:")]
pub async fn given_config_contains(_world: &mut World, step: &cucumber::gherkin::Step) {
    let docstring = step.docstring.as_ref().expect("Expected a docstring");
    tracing::debug!("Config contains: {}", docstring.trim());
}

// TEAM-049: Fixed quote handling using shell-aware parsing
#[when(expr = "I run {string}")]
pub async fn when_i_run_command_string(world: &mut World, command: String) {
    // TEAM-044: Execute the command for real, not just store it
    tracing::info!("🚀 Executing command: {}", command);

    // TEAM-049: Use shell-aware parsing to handle quotes properly
    let parts =
        shlex::split(&command).unwrap_or_else(|| panic!("Failed to parse command: {}", command));

    if parts.is_empty() {
        panic!("Empty command");
    }

    // Extract binary name and args
    let binary = &parts[0];
    let args: Vec<&str> = parts[1..].iter().map(|s| s.as_str()).collect();

    // Map command names to actual binary names
    let actual_binary = if binary == "rbee-keeper" { "rbee" } else { binary.as_str() };

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
// TEAM-048: Fixed multi-line command parsing (remove backslash line continuations)
// TEAM-049: Fixed quote handling using shell-aware parsing
#[when(expr = "I run:")]
pub async fn when_i_run_command_docstring(world: &mut World, step: &cucumber::gherkin::Step) {
    let docstring = step.docstring.as_ref().expect("Expected a docstring");

    // TEAM-048: Remove backslash line continuations (\ followed by newline and whitespace)
    let command_line = docstring
        .lines()
        .map(|line| line.trim_end_matches('\\').trim())
        .collect::<Vec<_>>()
        .join(" ");

    tracing::info!("🚀 Executing command: {}", command_line);

    // TEAM-049: Use shell-aware parsing to handle quotes properly
    // This fixes: --prompt "write a short story" being split incorrectly
    let parts = shlex::split(&command_line)
        .unwrap_or_else(|| panic!("Failed to parse command: {}", command_line));

    if parts.is_empty() {
        panic!("Empty command");
    }

    // Extract binary name and args
    let binary = &parts[0];
    let args: Vec<&str> = parts[1..].iter().map(|s| s.as_str()).collect();

    // TEAM-044: Map command names to actual binary names
    // rbee-keeper -> rbee (the actual binary name)
    let actual_binary = if binary == "rbee-keeper" { "rbee" } else { binary.as_str() };

    // TEAM-044: Use pre-built binaries instead of cargo run to avoid compilation timeouts
    let workspace_dir = std::env::var("CARGO_MANIFEST_DIR")
        .map(|p| std::path::PathBuf::from(p).parent().unwrap().parent().unwrap().to_path_buf())
        .unwrap_or_else(|_| std::path::PathBuf::from("/home/vince/Projects/llama-orch"));

    let binary_path = workspace_dir.join("target/debug").join(actual_binary);

    // TEAM-048: Debug logging
    tracing::debug!("Binary: {}", binary_path.display());
    tracing::debug!("Args: {:?}", args);

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
pub async fn when_rbee_config_set(_world: &mut World, path: String) {
    tracing::debug!("RBEE_CONFIG set to: {}", path);
}

#[when(expr = "RBEE_CONFIG is not set")]
pub async fn when_rbee_config_not_set(_world: &mut World) {
    tracing::debug!("RBEE_CONFIG is not set");
}

#[when(expr = "{string} exists")]
pub async fn when_file_exists(_world: &mut World, path: String) {
    tracing::debug!("File exists: {}", path);
}

#[when(expr = "neither RBEE_CONFIG nor user config exist")]
pub async fn when_neither_config_exists(_world: &mut World) {
    tracing::debug!("Neither RBEE_CONFIG nor user config exist");
}

#[when(expr = "rbee-keeper executes remote command on {string}")]
pub async fn when_execute_remote_command(_world: &mut World, node: String) {
    tracing::debug!("Executing remote command on: {}", node);
}

#[then(expr = "binaries are installed to {string}")]
pub async fn then_binaries_installed_to(_world: &mut World, path: String) {
    tracing::debug!("Binaries should be installed to: {}", path);
}

#[then(expr = "config directory is created at {string}")]
pub async fn then_config_dir_created(_world: &mut World, path: String) {
    tracing::debug!("Config directory should be created at: {}", path);
}

#[then(expr = "data directory is created at {string}")]
pub async fn then_data_dir_created(_world: &mut World, path: String) {
    tracing::debug!("Data directory should be created at: {}", path);
}

#[then(expr = "default config file is generated at {string}")]
pub async fn then_default_config_generated(_world: &mut World, path: String) {
    tracing::debug!("Default config should be generated at: {}", path);
}

#[then(expr = "the following binaries are copied:")]
pub async fn then_binaries_copied(_world: &mut World, step: &cucumber::gherkin::Step) {
    let table = step.table.as_ref().expect("Expected a data table");
    tracing::debug!("Binaries should be copied: {} entries", table.rows.len() - 1);
}

#[then(expr = "installation instructions are displayed")]
pub async fn then_installation_instructions(_world: &mut World) {
    tracing::debug!("Installation instructions should be displayed");
}

#[then(expr = "sudo permissions are required")]
pub async fn then_sudo_required(_world: &mut World) {
    tracing::debug!("Sudo permissions should be required");
}

#[then(expr = "rbee-keeper loads config from {string}")]
pub async fn then_load_config_from(_world: &mut World, path: String) {
    tracing::debug!("Should load config from: {}", path);
}

#[then(expr = "the command uses {string} instead of {string}")]
pub async fn then_command_uses_instead(_world: &mut World, actual: String, default: String) {
    tracing::debug!("Command should use '{}' instead of '{}'", actual, default);
}

#[then(expr = "git commands use {string} instead of {string}")]
pub async fn then_git_uses_instead(_world: &mut World, actual: String, default: String) {
    tracing::debug!("Git commands should use '{}' instead of '{}'", actual, default);
}

#[then(expr = "the command executes the full inference flow")]
pub async fn then_execute_full_flow(_world: &mut World) {
    tracing::debug!("Should execute full inference flow");
}

#[then(expr = "tokens are streamed to stdout")]
pub async fn then_tokens_streamed_stdout(_world: &mut World) {
    tracing::debug!("Tokens should be streamed to stdout");
}

#[given(expr = "workers are registered on multiple nodes")]
pub async fn given_workers_on_multiple_nodes(_world: &mut World) {
    tracing::debug!("Workers registered on multiple nodes");
}

#[given(regex = r#"^a worker with id "(.+)" is running$"#)]
pub async fn given_worker_with_id_running(world: &mut World, worker_id: String) {
    // TEAM-048: Start queen-rbee for worker shutdown tests
    tracing::info!("Starting queen-rbee for worker shutdown test");

    // Ensure queen-rbee is running (reuse topology setup)
    if world.queen_rbee_process.is_none() {
        crate::steps::beehive_registry::given_queen_rbee_running(world).await;
    }

    tracing::debug!("Worker {} is running (queen-rbee ready)", worker_id);
}

#[then(expr = "the output shows health status of workers on workstation")]
pub async fn then_output_shows_health_status(_world: &mut World) {
    tracing::debug!("Output should show health status");
}

#[then(expr = "logs from workstation are streamed to stdout")]
pub async fn then_logs_streamed(_world: &mut World) {
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

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Priority 14: Additional CLI Command Functions (TEAM-070)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// TEAM-070: Execute CLI command with arguments NICE!
#[when(expr = "I run the CLI command {string} with args {string}")]
pub async fn when_run_cli_command(world: &mut World, command: String, args: String) {
    tracing::info!("🚀 Executing CLI command: {} {}", command, args);

    // Parse arguments using shell-aware parsing
    let arg_parts = shlex::split(&args).unwrap_or_else(|| panic!("Failed to parse args: {}", args));

    // Map command names to actual binary names
    let actual_binary = if command == "rbee-keeper" { "rbee" } else { command.as_str() };

    // Get workspace directory
    let workspace_dir = std::env::var("CARGO_MANIFEST_DIR")
        .map(|p| std::path::PathBuf::from(p).parent().unwrap().parent().unwrap().to_path_buf())
        .unwrap_or_else(|_| std::path::PathBuf::from("/home/vince/Projects/llama-orch"));

    let binary_path = workspace_dir.join("target/debug").join(actual_binary);

    // Execute command
    let result = tokio::process::Command::new(&binary_path)
        .args(&arg_parts)
        .current_dir(&workspace_dir)
        .output()
        .await;

    match result {
        Ok(output) => {
            world.last_command = Some(format!("{} {}", command, args));
            world.last_exit_code = output.status.code();
            world.last_stdout = String::from_utf8_lossy(&output.stdout).to_string();
            world.last_stderr = String::from_utf8_lossy(&output.stderr).to_string();

            tracing::info!("✅ Command completed with exit code: {:?} NICE!", world.last_exit_code);
            if !world.last_stdout.is_empty() {
                tracing::info!("stdout: {}", world.last_stdout);
            }
            if !world.last_stderr.is_empty() {
                tracing::warn!("stderr: {}", world.last_stderr);
            }
        }
        Err(e) => {
            world.last_exit_code = Some(127); // Command not found
            world.last_stderr = format!("Failed to execute command: {}", e);
            tracing::warn!("⚠️  Command execution failed: {} NICE!", e);
        }
    }
}

// TEAM-070: Verify output contains expected text NICE!
#[then(expr = "the output contains {string}")]
pub async fn then_output_contains(world: &mut World, expected: String) {
    let combined_output = format!("{}\n{}", world.last_stdout, world.last_stderr);

    assert!(
        combined_output.contains(&expected),
        "Expected output to contain '{}', but got:\nstdout: {}\nstderr: {}",
        expected,
        world.last_stdout,
        world.last_stderr
    );

    tracing::info!("✅ Output contains '{}' NICE!", expected);
}

// TEAM-070: Verify exit code matches expected value NICE!
#[then(expr = "the command exits with code {int}")]
pub async fn then_command_exit_code(world: &mut World, expected_code: i32) {
    assert_eq!(
        world.last_exit_code,
        Some(expected_code),
        "Expected exit code {}, got {:?}",
        expected_code,
        world.last_exit_code
    );

    tracing::info!("✅ Command exited with code {} NICE!", expected_code);
}

// TEAM-112: Display output with docstring (commonly used in many tests)
#[then(expr = "rbee-keeper displays:")]
pub async fn then_keeper_displays(_world: &mut World, step: &cucumber::gherkin::Step) {
    let docstring = step.docstring.as_ref().expect("Expected a docstring");
    tracing::debug!("rbee-keeper should display: {}", docstring.trim());
    // TEAM-112: Stub implementation - actual validation would check world.last_stdout
}

// TEAM-112: Validation failure with specific message
#[then(expr = "validation fails with {string}")]
pub async fn then_validation_fails_with(world: &mut World, expected_message: String) {
    // TEAM-112: Check that command failed and error message contains expected text
    assert_eq!(world.last_exit_code, Some(1), "Expected validation to fail with exit code 1");
    assert!(
        world.last_stderr.contains(&expected_message)
            || world.last_stdout.contains(&expected_message),
        "Expected error message to contain '{}', but got:\nstderr: {}\nstdout: {}",
        expected_message,
        world.last_stderr,
        world.last_stdout
    );
    tracing::info!("✅ Validation failed with message: {}", expected_message);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEAM-118: Missing Steps (Batch 1)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// TEAM-123: REMOVED DUPLICATE - this step is already defined at line 261 as then_exit_code_is
// Duplicate step definitions cause cucumber to hang!

// Step 6: Configure rbee-keeper to spawn queen-rbee
#[given(expr = "rbee-keeper is configured to spawn queen-rbee")]
pub async fn given_keeper_configured_spawn_queen(world: &mut World) {
    world.keeper_config = Some("spawn_queen".to_string());
    tracing::info!("✅ rbee-keeper configured to spawn queen-rbee");
}

// TEAM-123: REMOVED DUPLICATE - real implementation in validation.rs:219
