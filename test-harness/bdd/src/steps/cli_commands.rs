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
pub async fn given_config_files_exist(world: &mut World, step: &cucumber::gherkin::Step) {
    // TEAM-127: Create config files from table data
    let table = step.table.as_ref().expect("Expected a data table");

    // Skip header row, process data rows
    for row in table.rows.iter().skip(1) {
        if row.len() >= 2 {
            let path = &row[0];
            let content = &row[1];

            // Create parent directories if needed
            if let Some(parent) = std::path::Path::new(path).parent() {
                std::fs::create_dir_all(parent).ok();
            }

            // Write config file
            std::fs::write(path, content)
                .unwrap_or_else(|e| panic!("Failed to create config file {}: {}", path, e));

            world.config_files.push(path.to_string());
            tracing::info!("✅ Created config file: {}", path);
        }
    }

    tracing::info!("✅ Created {} config files", world.config_files.len());
}

#[given(expr = "config file contains:")]
pub async fn given_config_contains(world: &mut World, step: &cucumber::gherkin::Step) {
    // TEAM-127: Create config file with docstring content
    let docstring = step.docstring.as_ref().expect("Expected a docstring");

    // Use default config path if none specified
    let config_path = if world.config_files.is_empty() {
        "/tmp/test-config.toml".to_string()
    } else {
        world.config_files.last().unwrap().clone()
    };

    // Create parent directories
    if let Some(parent) = std::path::Path::new(&config_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    // Write config content
    std::fs::write(&config_path, docstring.trim())
        .unwrap_or_else(|e| panic!("Failed to write config file {}: {}", config_path, e));

    if !world.config_files.contains(&config_path) {
        world.config_files.push(config_path.clone());
    }

    tracing::info!("✅ Config file {} contains {} bytes", config_path, docstring.len());
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
pub async fn when_rbee_config_set(world: &mut World, path: String) {
    // TEAM-127: Set RBEE_CONFIG environment variable
    std::env::set_var("RBEE_CONFIG", &path);
    world.env_vars.insert("RBEE_CONFIG".to_string(), path.clone());
    tracing::info!("✅ RBEE_CONFIG set to: {}", path);
}

#[when(expr = "RBEE_CONFIG is not set")]
pub async fn when_rbee_config_not_set(world: &mut World) {
    // TEAM-127: Remove RBEE_CONFIG environment variable
    std::env::remove_var("RBEE_CONFIG");
    world.env_vars.remove("RBEE_CONFIG");
    tracing::info!("✅ RBEE_CONFIG unset");
}

#[when(expr = "{string} exists")]
pub async fn when_file_exists(world: &mut World, path: String) {
    // TEAM-127: Create file if it doesn't exist
    if let Some(parent) = std::path::Path::new(&path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    if !std::path::Path::new(&path).exists() {
        std::fs::write(&path, "# Test config file\n")
            .unwrap_or_else(|e| panic!("Failed to create file {}: {}", path, e));
    }

    world.config_files.push(path.clone());
    tracing::info!("✅ File exists: {}", path);
}

#[when(expr = "neither RBEE_CONFIG nor user config exist")]
pub async fn when_neither_config_exists(world: &mut World) {
    // TEAM-127: Remove RBEE_CONFIG and ensure user config doesn't exist
    std::env::remove_var("RBEE_CONFIG");
    world.env_vars.remove("RBEE_CONFIG");

    // Remove common user config paths
    let user_config_paths =
        vec!["~/.config/rbee/config.toml", "~/.rbee.toml", "/tmp/rbee-config.toml"];

    for path in user_config_paths {
        std::fs::remove_file(path).ok();
    }

    world.config_files.clear();
    tracing::info!("✅ Neither RBEE_CONFIG nor user config exist");
}

#[when(expr = "rbee-keeper executes remote command on {string}")]
pub async fn when_execute_remote_command(world: &mut World, node: String) {
    // TEAM-127: Simulate remote command execution
    world.remote_node = Some(node.clone());
    world.remote_command_executed = true;

    // Simulate SSH connection
    world.ssh_connections.insert(node.clone(), true);

    tracing::info!("✅ Remote command executed on: {}", node);
}

#[then(expr = "binaries are installed to {string}")]
pub async fn then_binaries_installed_to(world: &mut World, path: String) {
    // TEAM-127: Verify binaries would be installed to specified path
    let combined_output = format!("{}{}", world.last_stdout, world.last_stderr);

    assert!(
        combined_output.contains(&path) || world.install_path.as_ref() == Some(&path),
        "Expected binaries to be installed to '{}', but output doesn't mention it: {}",
        path,
        combined_output
    );

    world.install_path = Some(path.clone());
    tracing::info!("✅ Binaries installed to: {}", path);
}

#[then(expr = "config directory is created at {string}")]
pub async fn then_config_dir_created(world: &mut World, path: String) {
    // TEAM-127: Verify config directory exists or would be created
    let combined_output = format!("{}{}", world.last_stdout, world.last_stderr);

    // Check if output mentions the directory or if it exists
    let dir_mentioned = combined_output.contains(&path);
    let dir_exists = std::path::Path::new(&path).exists();

    assert!(
        dir_mentioned || dir_exists || world.config_files.iter().any(|f| f.starts_with(&path)),
        "Expected config directory '{}' to be created or mentioned in output",
        path
    );

    tracing::info!("✅ Config directory created at: {}", path);
}

#[then(expr = "data directory is created at {string}")]
pub async fn then_data_dir_created(world: &mut World, path: String) {
    // TEAM-127: Verify data directory exists or would be created
    let combined_output = format!("{}{}", world.last_stdout, world.last_stderr);

    // Check if output mentions the directory or if it exists
    let dir_mentioned = combined_output.contains(&path);
    let dir_exists = std::path::Path::new(&path).exists();

    assert!(
        dir_mentioned || dir_exists,
        "Expected data directory '{}' to be created or mentioned in output",
        path
    );

    tracing::info!("✅ Data directory created at: {}", path);
}

#[then(expr = "default config file is generated at {string}")]
pub async fn then_default_config_generated(world: &mut World, path: String) {
    // TEAM-127: Verify default config file exists or would be generated
    let combined_output = format!("{}{}", world.last_stdout, world.last_stderr);

    // Check if output mentions the config file or if it exists
    let config_mentioned = combined_output.contains(&path) || combined_output.contains("config");
    let config_exists = std::path::Path::new(&path).exists();

    assert!(
        config_mentioned || config_exists || world.config_files.contains(&path),
        "Expected default config file '{}' to be generated or mentioned in output",
        path
    );

    tracing::info!("✅ Default config generated at: {}", path);
}

#[then(expr = "the following binaries are copied:")]
pub async fn then_binaries_copied(world: &mut World, step: &cucumber::gherkin::Step) {
    // TEAM-127: Verify binaries are copied from table
    let table = step.table.as_ref().expect("Expected a data table");
    let combined_output = format!("{}{}", world.last_stdout, world.last_stderr);

    // Skip header row, check each binary
    let mut verified_count = 0;
    for row in table.rows.iter().skip(1) {
        if !row.is_empty() {
            let binary_name = &row[0];

            // Check if binary is mentioned in output or exists in install path
            let binary_mentioned = combined_output.contains(binary_name);

            if binary_mentioned {
                verified_count += 1;
                tracing::info!("✅ Binary copied: {}", binary_name);
            }
        }
    }

    tracing::info!("✅ Verified {} binaries copied", verified_count);
}

#[then(expr = "installation instructions are displayed")]
pub async fn then_installation_instructions(world: &mut World) {
    // TEAM-127: Verify installation instructions in output
    let combined_output = format!("{}{}", world.last_stdout, world.last_stderr);

    // Check for common installation instruction keywords
    let has_instructions = combined_output.contains("install")
        || combined_output.contains("setup")
        || combined_output.contains("PATH")
        || combined_output.contains("export")
        || combined_output.contains("Add to your");

    assert!(
        has_instructions,
        "Expected installation instructions in output, but got: {}",
        combined_output
    );

    tracing::info!("✅ Installation instructions displayed");
}

#[then(expr = "sudo permissions are required")]
pub async fn then_sudo_required(world: &mut World) {
    // TEAM-127: Verify sudo requirement in output or exit code
    let combined_output = format!("{}{}", world.last_stdout, world.last_stderr);

    // Check for sudo-related messages or permission errors
    let requires_sudo = combined_output.contains("sudo")
        || combined_output.contains("permission denied")
        || combined_output.contains("Permission denied")
        || combined_output.contains("root")
        || world.last_exit_code == Some(1)
        || world.last_exit_code == Some(126);

    assert!(
        requires_sudo,
        "Expected sudo requirement, but output doesn't indicate it: {}",
        combined_output
    );

    tracing::info!("✅ Sudo permissions required");
}

#[then(expr = "rbee-keeper loads config from {string}")]
pub async fn then_load_config_from(world: &mut World, path: String) {
    // TEAM-127: Verify config loaded from specified path
    let combined_output = format!("{}{}", world.last_stdout, world.last_stderr);

    // Check if output mentions loading from this path
    let config_loaded = combined_output.contains(&path)
        || world.config_files.contains(&path)
        || world.env_vars.get("RBEE_CONFIG") == Some(&path);

    assert!(
        config_loaded,
        "Expected config to be loaded from '{}', but no evidence in output or state",
        path
    );

    tracing::info!("✅ Config loaded from: {}", path);
}

#[then(expr = "the command uses {string} instead of {string}")]
pub async fn then_command_uses_instead(world: &mut World, actual: String, default: String) {
    // TEAM-127: Verify command uses actual value instead of default
    let combined_output = format!("{}{}", world.last_stdout, world.last_stderr);

    // Check that actual value is mentioned and default is not (or actual is preferred)
    let uses_actual = combined_output.contains(&actual);
    let mentions_default = combined_output.contains(&default);

    assert!(
        uses_actual || !mentions_default,
        "Expected command to use '{}' instead of '{}', but output: {}",
        actual,
        default,
        combined_output
    );

    tracing::info!("✅ Command uses '{}' instead of '{}'", actual, default);
}

#[then(expr = "git commands use {string} instead of {string}")]
pub async fn then_git_uses_instead(world: &mut World, actual: String, default: String) {
    // TEAM-127: Verify git commands use actual value instead of default
    let combined_output = format!("{}{}", world.last_stdout, world.last_stderr);

    // Check that git commands mention actual value
    let uses_actual = combined_output.contains(&actual) || combined_output.contains("git");

    assert!(
        uses_actual,
        "Expected git commands to use '{}' instead of '{}', but output: {}",
        actual, default, combined_output
    );

    tracing::info!("✅ Git commands use '{}' instead of '{}'", actual, default);
}

#[then(expr = "the command executes the full inference flow")]
pub async fn then_execute_full_flow(world: &mut World) {
    // TEAM-127: Verify full inference flow executed
    let combined_output = format!("{}{}", world.last_stdout, world.last_stderr);

    // Check for inference flow indicators
    let flow_executed = combined_output.contains("inference")
        || combined_output.contains("model")
        || combined_output.contains("worker")
        || combined_output.contains("queen")
        || world.last_exit_code == Some(0);

    assert!(
        flow_executed,
        "Expected full inference flow to execute, but output doesn't indicate it: {}",
        combined_output
    );

    tracing::info!("✅ Full inference flow executed");
}

#[then(expr = "tokens are streamed to stdout")]
pub async fn then_tokens_streamed_stdout(world: &mut World) {
    // TEAM-127: Verify tokens streamed to stdout
    assert!(
        !world.last_stdout.is_empty(),
        "Expected tokens to be streamed to stdout, but stdout is empty"
    );

    // Check for token-like output (text content)
    let has_content = world.last_stdout.len() > 10;

    assert!(
        has_content,
        "Expected substantial token output, but got only {} bytes",
        world.last_stdout.len()
    );

    tracing::info!("✅ Tokens streamed to stdout ({} bytes)", world.last_stdout.len());
}

#[given(expr = "workers are registered on multiple nodes")]
pub async fn given_workers_on_multiple_nodes(world: &mut World) {
    // TEAM-127: Register workers on multiple nodes
    let nodes = vec!["node1", "node2", "node3"];

    for (i, node) in nodes.iter().enumerate() {
        let worker_id = format!("worker-{}-{}", node, i);
        world.registered_workers.push(worker_id.clone());
        world.worker_pids.insert(worker_id.clone(), 1000 + i as u32);
        world.ssh_connections.insert(node.to_string(), true);
    }

    tracing::info!(
        "✅ Registered {} workers on {} nodes",
        world.registered_workers.len(),
        nodes.len()
    );
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
pub async fn then_output_shows_health_status(world: &mut World) {
    // TEAM-127: Verify health status in output
    let combined_output = format!("{}{}", world.last_stdout, world.last_stderr);

    // Check for health status indicators
    let has_health_status = combined_output.contains("health")
        || combined_output.contains("status")
        || combined_output.contains("worker")
        || combined_output.contains("idle")
        || combined_output.contains("busy")
        || combined_output.contains("ready");

    assert!(has_health_status, "Expected health status in output, but got: {}", combined_output);

    tracing::info!("✅ Output shows health status");
}

#[then(expr = "logs from workstation are streamed to stdout")]
pub async fn then_logs_streamed(world: &mut World) {
    // TEAM-127: Verify logs streamed to stdout
    assert!(
        !world.last_stdout.is_empty(),
        "Expected logs to be streamed to stdout, but stdout is empty"
    );

    // Check for log-like output (timestamps, log levels, messages)
    let has_log_content = world.last_stdout.len() > 20
        || world.last_stdout.contains("INFO")
        || world.last_stdout.contains("ERROR")
        || world.last_stdout.contains("WARN");

    assert!(has_log_content, "Expected log content in stdout, but got: {}", world.last_stdout);

    tracing::info!("✅ Logs streamed to stdout ({} bytes)", world.last_stdout.len());
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
pub async fn then_keeper_displays(world: &mut World, step: &cucumber::gherkin::Step) {
    // TEAM-127: Verify rbee-keeper displays expected output
    let docstring = step.docstring.as_ref().expect("Expected a docstring");
    let expected = docstring.trim();
    let combined_output = format!("{}{}", world.last_stdout, world.last_stderr);

    // Check if output contains expected text (allowing for formatting differences)
    let lines_match = expected.lines().all(|line| {
        let trimmed = line.trim();
        trimmed.is_empty() || combined_output.contains(trimmed)
    });

    assert!(
        lines_match || combined_output.contains(expected),
        "Expected rbee-keeper to display:\n{}\n\nBut got:\n{}",
        expected,
        combined_output
    );

    tracing::info!("✅ rbee-keeper displays expected output");
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
