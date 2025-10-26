// TEAM-301: Integration tests for process output streaming
//
// Tests that keeper correctly displays daemon output to the terminal.

use rbee_keeper::process_utils::spawn_with_output_streaming;
use tokio::process::Command;

#[tokio::test]
async fn test_spawn_with_echo_output() {
    let mut command = Command::new("echo");
    command.arg("Test output from process");
    
    let mut child = spawn_with_output_streaming(command).await
        .expect("Failed to spawn command");
    
    // Wait for process to complete
    let status = child.wait().await.expect("Failed to wait for process");
    assert!(status.success(), "Process should exit successfully");
}

#[tokio::test]
async fn test_spawn_with_multiple_lines() {
    let mut command = Command::new("sh");
    command.arg("-c");
    command.arg("echo 'Line 1' && echo 'Line 2' && echo 'Line 3'");
    
    let mut child = spawn_with_output_streaming(command).await
        .expect("Failed to spawn command");
    
    // Wait for process to complete
    let status = child.wait().await.expect("Failed to wait for process");
    assert!(status.success(), "Process should exit successfully");
}

#[tokio::test]
async fn test_spawn_with_stderr_output() {
    let mut command = Command::new("sh");
    command.arg("-c");
    command.arg("echo 'Error message' >&2");
    
    let mut child = spawn_with_output_streaming(command).await
        .expect("Failed to spawn command");
    
    // Wait for process to complete
    let status = child.wait().await.expect("Failed to wait for process");
    assert!(status.success(), "Process should exit successfully");
}

#[tokio::test]
async fn test_spawn_with_mixed_output() {
    let mut command = Command::new("sh");
    command.arg("-c");
    command.arg("echo 'stdout' && echo 'stderr' >&2 && echo 'stdout again'");
    
    let mut child = spawn_with_output_streaming(command).await
        .expect("Failed to spawn command");
    
    // Wait for process to complete
    let status = child.wait().await.expect("Failed to wait for process");
    assert!(status.success(), "Process should exit successfully");
}

#[tokio::test]
async fn test_spawn_nonexistent_command() {
    let command = Command::new("this-command-does-not-exist-12345");
    
    let result = spawn_with_output_streaming(command).await;
    assert!(result.is_err(), "Should fail to spawn nonexistent command");
}

#[tokio::test]
async fn test_spawn_command_with_error_exit() {
    let mut command = Command::new("sh");
    command.arg("-c");
    command.arg("echo 'Starting...' && exit 1");
    
    let mut child = spawn_with_output_streaming(command).await
        .expect("Failed to spawn command");
    
    // Wait for process to complete
    let status = child.wait().await.expect("Failed to wait for process");
    assert!(!status.success(), "Process should exit with error code");
}

#[tokio::test]
async fn test_spawn_with_narration_format_output() {
    // Simulates daemon output with narration format
    let mut command = Command::new("echo");
    command.arg("[queen     ] startup         : Starting queen-rbee on port 8500");
    
    let mut child = spawn_with_output_streaming(command).await
        .expect("Failed to spawn command");
    
    // Wait for process to complete
    let status = child.wait().await.expect("Failed to wait for process");
    assert!(status.success(), "Process should exit successfully");
}

#[tokio::test]
async fn test_spawn_long_running_process() {
    // Simulates a daemon that runs for a short time
    let mut command = Command::new("sh");
    command.arg("-c");
    command.arg("echo 'Starting...' && sleep 0.1 && echo 'Ready'");
    
    let mut child = spawn_with_output_streaming(command).await
        .expect("Failed to spawn command");
    
    // Wait for process to complete
    let status = child.wait().await.expect("Failed to wait for process");
    assert!(status.success(), "Process should exit successfully");
}
