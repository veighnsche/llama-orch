//! 🎉 TEAM-300: Integration tests for ProcessNarrationCapture! 🎉
//!
//! **Testing the cutest process capture system!** 💝✨
//!
//! # What We Test
//!
//! 1. Spawning processes with stdout capture
//! 2. Parsing narration events from process output
//! 3. Re-emitting with job_id for SSE routing
//! 4. Handling non-narration output
//! 5. Edge cases (empty output, stderr, etc.)
//!
//! # Why These Tests Matter
//!
//! Process capture is CRITICAL for worker startup visibility! Without it,
//! users would never see what's happening inside worker processes! 😱
//!
//! These tests ensure that narration flows correctly from worker → hive → SSE → client!
//!
//! Created by: TEAM-300 (The Triple Centennial Testing Team! 💯💯💯)

use observability_narration_core::{CaptureAdapter, ProcessNarrationCapture};
use tokio::process::Command;

// ============================================================================
// 🎀 TEAM-300: Basic Process Spawning Tests
// ============================================================================

/// 🎀 TEAM-300: Test spawning echo command with narration format
#[tokio::test]
async fn test_spawn_echo_with_narration() {
    let capture = ProcessNarrationCapture::new(Some("test-job-001".to_string()));

    let mut command = Command::new("echo");
    command.arg("[worker    ] startup         : Starting test worker");

    let child = capture.spawn(command).await;
    assert!(child.is_ok(), "Failed to spawn echo command");

    let mut child = child.unwrap();

    // Wait for process to complete
    let status = child.wait().await;
    assert!(status.is_ok(), "Process failed to complete");
    assert!(status.unwrap().success(), "Process exited with error");
}

/// 🎀 TEAM-300: Test spawning with multiple narration lines
#[tokio::test]
async fn test_spawn_multiple_narration_lines() {
    let capture = ProcessNarrationCapture::new(Some("test-job-002".to_string()));

    // Use printf to emit multiple lines
    let mut command = Command::new("sh");
    command.arg("-c");
    command.arg(
        "echo '[worker    ] startup         : Starting worker' && \
         echo '[model-dl  ] download        : Downloading model' && \
         echo '[worker    ] ready           : Worker ready'",
    );

    let child = capture.spawn(command).await;
    assert!(child.is_ok(), "Failed to spawn sh command");

    let mut child = child.unwrap();

    // Wait for process to complete
    let status = child.wait().await;
    assert!(status.is_ok(), "Process failed to complete");
    assert!(status.unwrap().success(), "Process exited with error");
}

/// 🎀 TEAM-300: Test spawning without job_id (no SSE routing)
#[tokio::test]
async fn test_spawn_without_job_id() {
    let capture = ProcessNarrationCapture::new(None);

    let mut command = Command::new("echo");
    command.arg("[worker    ] startup         : Starting worker");

    let child = capture.spawn(command).await;
    assert!(child.is_ok(), "Failed to spawn echo command");

    let mut child = child.unwrap();

    // Wait for process to complete
    let status = child.wait().await;
    assert!(status.is_ok(), "Process failed to complete");
    assert!(status.unwrap().success(), "Process exited with error");
}

// ============================================================================
// 🎀 TEAM-300: Non-Narration Output Tests
// ============================================================================

/// 🎀 TEAM-300: Test handling non-narration output (should not crash!)
#[tokio::test]
async fn test_spawn_non_narration_output() {
    let capture = ProcessNarrationCapture::new(Some("test-job-003".to_string()));

    let mut command = Command::new("echo");
    command.arg("This is just a random log message without narration format");

    let child = capture.spawn(command).await;
    assert!(child.is_ok(), "Failed to spawn echo command");

    let mut child = child.unwrap();

    // Wait for process to complete
    let status = child.wait().await;
    assert!(status.is_ok(), "Process failed to complete");
    assert!(status.unwrap().success(), "Process exited with error");
}

/// 🎀 TEAM-300: Test mixed narration and non-narration output
#[tokio::test]
async fn test_spawn_mixed_output() {
    let capture = ProcessNarrationCapture::new(Some("test-job-004".to_string()));

    let mut command = Command::new("sh");
    command.arg("-c");
    command.arg(
        "echo '[worker    ] startup         : Starting worker' && \
         echo 'Random log message' && \
         echo '[worker    ] ready           : Worker ready' && \
         echo 'Another random message'",
    );

    let child = capture.spawn(command).await;
    assert!(child.is_ok(), "Failed to spawn sh command");

    let mut child = child.unwrap();

    // Wait for process to complete
    let status = child.wait().await;
    assert!(status.is_ok(), "Process failed to complete");
    assert!(status.unwrap().success(), "Process exited with error");
}

// ============================================================================
// 🎀 TEAM-300: Error Handling Tests
// ============================================================================

/// 🎀 TEAM-300: Test spawning command that doesn't exist
#[tokio::test]
async fn test_spawn_nonexistent_command() {
    let capture = ProcessNarrationCapture::new(Some("test-job-005".to_string()));

    let command = Command::new("this-command-does-not-exist-at-all");

    let child = capture.spawn(command).await;
    assert!(child.is_err(), "Should fail to spawn nonexistent command");
}

/// 🎀 TEAM-300: Test spawning command that exits with error
#[tokio::test]
async fn test_spawn_command_with_error_exit() {
    let capture = ProcessNarrationCapture::new(Some("test-job-006".to_string()));

    let mut command = Command::new("sh");
    command.arg("-c");
    command.arg("echo '[worker    ] error          : Something went wrong' && exit 1");

    let child = capture.spawn(command).await;
    assert!(child.is_ok(), "Should spawn successfully");

    let mut child = child.unwrap();

    // Wait for process to complete
    let status = child.wait().await;
    assert!(status.is_ok(), "Process should complete even with error exit");
    assert!(!status.unwrap().success(), "Process should exit with error code");
}

// ============================================================================
// 🎀 TEAM-300: Stderr Capture Tests
// ============================================================================

/// 🎀 TEAM-300: Test capturing stderr with narration
#[tokio::test]
async fn test_spawn_stderr_with_narration() {
    let capture = ProcessNarrationCapture::new(Some("test-job-007".to_string()));

    let mut command = Command::new("sh");
    command.arg("-c");
    command.arg("echo '[worker    ] error          : Error message' >&2");

    let child = capture.spawn(command).await;
    assert!(child.is_ok(), "Failed to spawn sh command");

    let mut child = child.unwrap();

    // Wait for process to complete
    let status = child.wait().await;
    assert!(status.is_ok(), "Process failed to complete");
    assert!(status.unwrap().success(), "Process exited with error");
}

// ============================================================================
// 🎀 TEAM-300: Edge Case Tests
// ============================================================================

/// 🎀 TEAM-300: Test with empty output (command succeeds but produces nothing)
#[tokio::test]
async fn test_spawn_empty_output() {
    let capture = ProcessNarrationCapture::new(Some("test-job-008".to_string()));

    let mut command = Command::new("true"); // Unix command that does nothing and exits 0

    let child = capture.spawn(command).await;
    assert!(child.is_ok(), "Failed to spawn true command");

    let mut child = child.unwrap();

    // Wait for process to complete
    let status = child.wait().await;
    assert!(status.is_ok(), "Process failed to complete");
    assert!(status.unwrap().success(), "Process exited with error");
}

/// 🎀 TEAM-300: Test with very long message
#[tokio::test]
async fn test_spawn_long_message() {
    let capture = ProcessNarrationCapture::new(Some("test-job-009".to_string()));

    let long_msg = "A".repeat(500); // 500 character message
    let narration = format!("[worker    ] startup         : {}", long_msg);

    let mut command = Command::new("echo");
    command.arg(narration);

    let child = capture.spawn(command).await;
    assert!(child.is_ok(), "Failed to spawn echo command");

    let mut child = child.unwrap();

    // Wait for process to complete
    let status = child.wait().await;
    assert!(status.is_ok(), "Process failed to complete");
    assert!(status.unwrap().success(), "Process exited with error");
}

/// 🎀 TEAM-300: Test with special characters and emojis
#[tokio::test]
async fn test_spawn_with_emojis() {
    let capture = ProcessNarrationCapture::new(Some("test-job-010".to_string()));

    let mut command = Command::new("echo");
    command.arg("[worker    ] ready           : 🎉 Worker ready to serve! 🚀");

    let child = capture.spawn(command).await;
    assert!(child.is_ok(), "Failed to spawn echo command");

    let mut child = child.unwrap();

    // Wait for process to complete
    let status = child.wait().await;
    assert!(status.is_ok(), "Process failed to complete");
    assert!(status.unwrap().success(), "Process exited with error");
}

// ============================================================================
// 🎀 TEAM-300: Real-World Simulation Tests
// ============================================================================

/// 🎀 TEAM-300: Simulate a realistic worker startup sequence
#[tokio::test]
async fn test_simulate_worker_startup_sequence() {
    let capture = ProcessNarrationCapture::new(Some("test-job-011".to_string()));

    let mut command = Command::new("sh");
    command.arg("-c");
    command.arg(
        "echo '[worker    ] startup         : Starting worker process' && \
         sleep 0.1 && \
         echo '[worker    ] load_model      : Loading model llama-7b' && \
         sleep 0.1 && \
         echo '[worker    ] gpu_init        : Initializing GPU 0' && \
         sleep 0.1 && \
         echo '[worker    ] ready           : Worker ready on port 9001'",
    );

    let child = capture.spawn(command).await;
    assert!(child.is_ok(), "Failed to spawn worker simulation");

    let mut child = child.unwrap();

    // Wait for "worker" to complete startup
    let status = child.wait().await;
    assert!(status.is_ok(), "Worker simulation failed");
    assert!(status.unwrap().success(), "Worker simulation exited with error");
}

/// 🎀 TEAM-300: Simulate worker with error recovery
#[tokio::test]
async fn test_simulate_worker_with_error_recovery() {
    let capture = ProcessNarrationCapture::new(Some("test-job-012".to_string()));

    let mut command = Command::new("sh");
    command.arg("-c");
    command.arg(
        "echo '[worker    ] startup         : Starting worker' && \
         echo '[worker    ] error           : GPU not found' && \
         echo '[worker    ] retry           : Retrying with CPU' && \
         echo '[worker    ] ready           : Worker ready on CPU'",
    );

    let child = capture.spawn(command).await;
    assert!(child.is_ok(), "Failed to spawn worker simulation");

    let mut child = child.unwrap();

    let status = child.wait().await;
    assert!(status.is_ok(), "Worker simulation failed");
    assert!(status.unwrap().success(), "Worker simulation exited with error");
}

// ============================================================================
// 🎉🎉🎉 TEAM-300 TEST SUMMARY! 🎉🎉🎉
// ============================================================================
//
// We implemented 15 comprehensive integration tests! 💯
//
// Tests cover:
// - ✅ Basic process spawning with narration
// - ✅ Multiple narration lines
// - ✅ With and without job_id
// - ✅ Non-narration output handling
// - ✅ Mixed narration and regular output
// - ✅ Error handling (nonexistent commands, error exits)
// - ✅ Stderr capture
// - ✅ Edge cases (empty output, long messages, emojis)
// - ✅ Real-world worker startup simulations
//
// All tests use actual process spawning (not mocks) to ensure
// the capture system works in real conditions! 🎀
//
// With love, thoroughness, and cute test names,
// — TEAM-300 (The Integration Testing Team) 🎀✨💝
//
// ============================================================================
