// TEAM-251: Assertions library for integration tests
// Purpose: Reusable assertions for command results and state validation

use crate::integration::harness::{CommandResult, TestHarness};

/// Assert command succeeded (exit code 0)
pub fn assert_success(result: &CommandResult) {
    assert_eq!(
        result.exit_code,
        Some(0),
        "Command failed with exit code {:?}\nStdout: {}\nStderr: {}",
        result.exit_code,
        result.stdout,
        result.stderr
    );
}

/// Assert command failed (exit code != 0)
pub fn assert_failure(result: &CommandResult) {
    assert_ne!(
        result.exit_code,
        Some(0),
        "Command should have failed but succeeded\nStdout: {}\nStderr: {}",
        result.stdout,
        result.stderr
    );
}

/// Assert output contains text (stdout or stderr)
pub fn assert_output_contains(result: &CommandResult, text: &str) {
    let combined = format!("{}\n{}", result.stdout, result.stderr);
    assert!(
        combined.contains(text),
        "Output should contain '{}'\nStdout: {}\nStderr: {}",
        text,
        result.stdout,
        result.stderr
    );
}

/// Assert stdout contains text
pub fn assert_stdout_contains(result: &CommandResult, text: &str) {
    assert!(
        result.stdout.contains(text),
        "Stdout should contain '{}'\nActual stdout: {}",
        text,
        result.stdout
    );
}

/// Assert stderr contains text
pub fn assert_stderr_contains(result: &CommandResult, text: &str) {
    assert!(
        result.stderr.contains(text),
        "Stderr should contain '{}'\nActual stderr: {}",
        text,
        result.stderr
    );
}

/// Assert process is running
pub fn assert_running(harness: &TestHarness, name: &str) {
    assert!(
        harness.is_running(name),
        "{} should be running but is not",
        name
    );
}

/// Assert process is stopped
pub fn assert_stopped(harness: &TestHarness, name: &str) {
    assert!(
        !harness.is_running(name),
        "{} should be stopped but is running",
        name
    );
}

/// Assert exit code matches expected
pub fn assert_exit_code(result: &CommandResult, expected: i32) {
    assert_eq!(
        result.exit_code,
        Some(expected),
        "Expected exit code {} but got {:?}\nStdout: {}\nStderr: {}",
        expected,
        result.exit_code,
        result.stdout,
        result.stderr
    );
}

/// Assert output is empty
pub fn assert_output_empty(result: &CommandResult) {
    assert!(
        result.stdout.trim().is_empty() && result.stderr.trim().is_empty(),
        "Output should be empty\nStdout: {}\nStderr: {}",
        result.stdout,
        result.stderr
    );
}

/// Assert stdout is empty
pub fn assert_stdout_empty(result: &CommandResult) {
    assert!(
        result.stdout.trim().is_empty(),
        "Stdout should be empty\nActual stdout: {}",
        result.stdout
    );
}

/// Assert stderr is empty
pub fn assert_stderr_empty(result: &CommandResult) {
    assert!(
        result.stderr.trim().is_empty(),
        "Stderr should be empty\nActual stderr: {}",
        result.stderr
    );
}
