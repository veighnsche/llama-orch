// TEAM-111: Core test execution logic

use super::types::{BddConfig, OutputPaths, TestResults};
use super::{files, parser, reporter};
use anyhow::{Context, Result};
use chrono::Local;
use colored::Colorize;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Main entry point for BDD test execution
pub fn run_bdd_tests(config: BddConfig) -> Result<()> {
    // Start timer
    let start_time = std::time::Instant::now();

    // Validate cargo is available
    validate_cargo_available()?;

    let bdd_dir = get_bdd_directory()?;
    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();

    // Setup logging
    let log_dir = bdd_dir.join(".test-logs");
    fs::create_dir_all(&log_dir)
        .context(format!("Failed to create log directory: {}", log_dir.display()))?;
    let mut paths = OutputPaths::new(log_dir, &timestamp);

    // Print banner
    reporter::print_banner(&config, &timestamp);

    // Phase 1: Validate environment
    validate_environment(&bdd_dir)?;

    // Phase 2: Compilation check
    println!("\n{} {}", "[1/4]".yellow(), "Checking compilation...".cyan());
    check_compilation(&bdd_dir, &paths, config.quiet)?;
    println!("{}", "✅ Compilation successful".green());

    // Phase 3: Discover test scenarios
    println!("\n{} {}", "[2/4]".yellow(), "Discovering test scenarios...".cyan());
    let scenario_count = discover_scenarios(&bdd_dir)?;
    println!(
        "{} {}",
        "📊 Found".green(),
        format!("{} scenarios in feature files", scenario_count).green()
    );

    // Phase 4: Run tests
    println!("\n{} {}", "[3/4]".yellow(), "Running BDD tests...".cyan());

    // Check if we should run only failing tests
    let test_cmd = if config.run_all {
        println!("{}", "🔄 Running ALL tests (--all flag specified)".yellow());
        build_test_command(&config)
    } else {
        // Try to find the last rerun command
        match find_last_rerun_command(&paths.log_dir) {
            Some(rerun_cmd) => {
                println!(
                    "{}",
                    "⚡ Running ONLY failing tests from last run (default behavior)".green().bold()
                );
                println!("{}", "💡 Use --all to run all tests".cyan());
                println!();
                rerun_cmd
            }
            None => {
                println!("{}", "📝 No previous failures found - running ALL tests".yellow());
                build_test_command(&config)
            }
        }
    };

    println!("{} {}", "Command:".blue(), test_cmd);

    let results = execute_tests(&bdd_dir, &test_cmd, &paths, config.quiet)?;

    // Phase 5: Parse and report results
    println!("\n{} {}", "[4/4]".yellow(), "Parsing test results...".cyan());

    // Calculate elapsed time
    let elapsed = start_time.elapsed();

    reporter::print_test_summary(&results, elapsed);

    // Handle failures if any
    if results.failed > 0 {
        paths.set_failure_files(&timestamp);
        files::generate_failure_files(&paths, &results)?;
        reporter::print_failure_details(&paths)?;
    }

    // Generate summary file
    files::generate_summary_file(&paths, &test_cmd, &results)?;

    // Display output files
    reporter::print_output_files(&paths, results.failed > 0, results.failed, elapsed);

    // Final banner
    reporter::print_final_banner(results.exit_code == 0);

    // Show quiet warning if needed
    if config.show_quiet_warning {
        reporter::print_quiet_warning();
    }

    std::process::exit(results.exit_code)
}

fn get_bdd_directory() -> Result<PathBuf> {
    let root = crate::util::repo_root()?;
    Ok(root.join("test-harness/bdd"))
}

fn find_last_rerun_command(log_dir: &PathBuf) -> Option<String> {
    // Look for the rerun-failures-cmd.txt file
    let rerun_file = log_dir.join("rerun-failures-cmd.txt");

    if !rerun_file.exists() {
        return None;
    }

    // Read the command from the file
    match fs::read_to_string(&rerun_file) {
        Ok(content) => {
            let content = content.trim();
            if content.is_empty() {
                None
            } else {
                Some(content.to_string())
            }
        }
        Err(_) => None,
    }
}

fn validate_cargo_available() -> Result<()> {
    Command::new("cargo")
        .arg("--version")
        .output()
        .context("Failed to execute 'cargo --version'. Is cargo installed and in PATH?")?;
    Ok(())
}

fn validate_environment(bdd_dir: &PathBuf) -> Result<()> {
    if !bdd_dir.join("Cargo.toml").exists() {
        anyhow::bail!("Cannot find Cargo.toml in {}", bdd_dir.display());
    }

    if !bdd_dir.join("tests/features").exists() {
        println!("{} No tests/features directory found", "⚠️  WARNING:".yellow());
    }

    Ok(())
}

fn check_compilation(bdd_dir: &PathBuf, paths: &OutputPaths, quiet: bool) -> Result<()> {
    let mut cmd = Command::new("cargo");
    cmd.arg("check").arg("--lib").current_dir(bdd_dir);

    if quiet {
        let output = cmd.output().context("running cargo check")?;
        fs::write(&paths.compile_log, &output.stdout)?;
        fs::write(&paths.compile_log, &output.stderr)?;

        if !output.status.success() {
            anyhow::bail!("Compilation failed! Check {}", paths.compile_log.display());
        }
    } else {
        let status = cmd.status().context("running cargo check")?;
        if !status.success() {
            anyhow::bail!("Compilation failed!");
        }
    }

    Ok(())
}

fn discover_scenarios(bdd_dir: &PathBuf) -> Result<usize> {
    let features_dir = bdd_dir.join("tests/features");
    if !features_dir.exists() {
        return Ok(0);
    }

    let mut count = 0;
    for entry in walkdir::WalkDir::new(features_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("feature"))
    {
        let content = fs::read_to_string(entry.path())?;
        count += content.lines().filter(|line| line.trim().starts_with("Scenario:")).count();
    }

    Ok(count)
}

fn build_test_command(config: &BddConfig) -> String {
    let mut cmd = "cargo test --test cucumber".to_string();

    if let Some(ref tags) = config.tags {
        cmd.push_str(&format!(" -- --tags {}", tags));
    }

    if let Some(ref feature) = config.feature {
        cmd.push_str(&format!(" -- {}", feature));
    }

    cmd
}

fn execute_tests(
    bdd_dir: &PathBuf,
    test_cmd: &str,
    paths: &OutputPaths,
    quiet: bool,
) -> Result<TestResults> {
    reporter::print_test_execution_start();

    let parts: Vec<&str> = test_cmd.split_whitespace().collect();
    let mut cmd = Command::new(parts[0]);
    for arg in &parts[1..] {
        cmd.arg(arg);
    }
    cmd.current_dir(bdd_dir);

    let results = if quiet {
        execute_tests_quiet(&mut cmd, paths)?
    } else {
        execute_tests_live(&mut cmd, paths)?
    };

    reporter::print_test_execution_end();

    Ok(results)
}

fn execute_tests_quiet(cmd: &mut Command, paths: &OutputPaths) -> Result<TestResults> {
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .unwrap()
            .tick_strings(&["-", "\\", "|", "/"]),
    );
    pb.set_message("Running tests...");
    pb.enable_steady_tick(Duration::from_millis(100));

    let output = cmd.output().context("Failed to execute test command")?;

    pb.finish_with_message("Running tests... Done!");

    // Combine stdout and stderr
    let mut combined_output = output.stdout.clone();
    combined_output.extend_from_slice(&output.stderr);

    // Save output with error handling
    fs::write(&paths.test_output, &combined_output)
        .context(format!("Failed to write test output to {}", paths.test_output.display()))?;

    // Parse results - handle invalid UTF-8 gracefully
    let output_str = String::from_utf8_lossy(&combined_output);
    let results = parser::parse_test_output(&output_str, output.status.code().unwrap_or(1));

    Ok(results)
}

fn execute_tests_live(cmd: &mut Command, paths: &OutputPaths) -> Result<TestResults> {
    println!("{}", "📺 LIVE OUTPUT MODE - You will see ALL test output below:".green());
    println!();

    // TEAM-111: Pipe output and read concurrently to avoid deadlock
    cmd.stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = cmd.spawn().context("spawning test process")?;
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    // Shared output buffer
    let output_content = Arc::new(Mutex::new(Vec::new()));
    let output_content_stdout = Arc::clone(&output_content);
    let output_content_stderr = Arc::clone(&output_content);

    // Spawn thread for stdout
    let stdout_handle = thread::spawn(move || {
        let reader = BufReader::new(stdout);
        for line in reader.lines().flatten() {
            println!("{}", line);
            output_content_stdout.lock().unwrap().push(line);
        }
    });

    // Spawn thread for stderr
    let stderr_handle = thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines().flatten() {
            eprintln!("{}", line);
            output_content_stderr.lock().unwrap().push(line);
        }
    });

    // Wait for both threads to finish - handle panics gracefully
    if let Err(e) = stdout_handle.join() {
        eprintln!("Warning: stdout reader thread panicked: {:?}", e);
    }
    if let Err(e) = stderr_handle.join() {
        eprintln!("Warning: stderr reader thread panicked: {:?}", e);
    }

    // Wait for process to complete
    let status = child.wait()?;

    // Get captured output - handle poisoned mutex
    let output_str = match output_content.lock() {
        Ok(lines) => lines.join("\n"),
        Err(poisoned) => {
            eprintln!("Warning: output buffer mutex was poisoned, recovering data");
            poisoned.into_inner().join("\n")
        }
    };

    // Save to file with better error handling
    fs::write(&paths.test_output, &output_str)
        .context(format!("Failed to write test output to {}", paths.test_output.display()))?;

    // Parse results
    let results = parser::parse_test_output(&output_str, status.code().unwrap_or(1));

    Ok(results)
}
