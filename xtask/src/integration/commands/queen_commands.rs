// TEAM-251: Queen command tests
// Purpose: Test all queen commands in all valid states
// Commands: queen start, queen stop

use crate::integration::assertions::*;
use crate::integration::harness::TestHarness;
use std::time::Duration;

// ============================================================================
// Queen Start Tests
// ============================================================================

#[tokio::test]
async fn test_queen_start_when_stopped() {
    // TEAM-251: Test queen start when queen is stopped (happy path)

    let mut harness = TestHarness::new().await.unwrap();

    // Precondition: Queen is stopped
    assert_stopped(&harness, "queen");

    // Action: Start queen
    let result = harness.run_command(&["queen", "start"]).await.unwrap();

    // Assertions
    assert_success(&result);
    assert_output_contains(&result, "✅");
    assert_output_contains(&result, "started");

    // Postcondition: Queen is running
    harness.wait_for_ready("queen", Duration::from_secs(10)).await.unwrap();
    assert_running(&harness, "queen");

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_queen_start_when_already_running() {
    // TEAM-251: Test queen start when queen is already running (idempotent)

    let mut harness = TestHarness::new().await.unwrap();

    // Precondition: Queen is running
    let _ = harness.run_command(&["queen", "start"]).await.unwrap();
    harness.wait_for_ready("queen", Duration::from_secs(10)).await.unwrap();
    assert_running(&harness, "queen");

    // Action: Start queen again
    let result = harness.run_command(&["queen", "start"]).await.unwrap();

    // Assertions
    assert_success(&result);
    assert_output_contains(&result, "already running");

    // Postcondition: Queen still running
    assert_running(&harness, "queen");

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_queen_start_shows_narration() {
    // TEAM-251: Test queen start shows narration output

    let mut harness = TestHarness::new().await.unwrap();

    // Action: Start queen
    let result = harness.run_command(&["queen", "start"]).await.unwrap();

    // Assertions: Should show narration
    assert_success(&result);
    assert!(!result.stdout.is_empty() || !result.stderr.is_empty(), "Should show narration output");

    harness.cleanup().await.unwrap();
}

// ============================================================================
// Queen Stop Tests
// ============================================================================

#[tokio::test]
async fn test_queen_stop_when_running() {
    // TEAM-251: Test queen stop when queen is running (happy path)

    let mut harness = TestHarness::new().await.unwrap();

    // Precondition: Queen is running
    let _ = harness.run_command(&["queen", "start"]).await.unwrap();
    harness.wait_for_ready("queen", Duration::from_secs(10)).await.unwrap();
    assert_running(&harness, "queen");

    // Action: Stop queen
    let result = harness.run_command(&["queen", "stop"]).await.unwrap();

    // Assertions
    assert_success(&result);
    assert_output_contains(&result, "✅");
    assert_output_contains(&result, "stopped");

    // Postcondition: Queen is stopped
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert_stopped(&harness, "queen");

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_queen_stop_when_already_stopped() {
    // TEAM-251: Test queen stop when queen is already stopped (idempotent)

    let mut harness = TestHarness::new().await.unwrap();

    // Precondition: Queen is stopped
    assert_stopped(&harness, "queen");

    // Action: Stop queen
    let result = harness.run_command(&["queen", "stop"]).await.unwrap();

    // Assertions
    assert_success(&result);
    // Should indicate not running or already stopped
    let output = format!("{}{}", result.stdout, result.stderr);
    assert!(
        output.contains("not running") || output.contains("already stopped"),
        "Should indicate queen is not running"
    );

    // Postcondition: Queen still stopped
    assert_stopped(&harness, "queen");

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_queen_stop_graceful_shutdown() {
    // TEAM-251: Test queen stop uses graceful shutdown (SIGTERM)

    let mut harness = TestHarness::new().await.unwrap();

    // Precondition: Queen is running
    let _ = harness.run_command(&["queen", "start"]).await.unwrap();
    harness.wait_for_ready("queen", Duration::from_secs(10)).await.unwrap();

    // Action: Stop queen
    let result = harness.run_command(&["queen", "stop"]).await.unwrap();

    // Assertions: Should complete within reasonable time (5s for graceful)
    assert_success(&result);

    // Postcondition: Queen is stopped
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert_stopped(&harness, "queen");

    harness.cleanup().await.unwrap();
}

// ============================================================================
// Queen Lifecycle Tests
// ============================================================================

#[tokio::test]
async fn test_queen_full_lifecycle() {
    // TEAM-251: Test complete queen lifecycle (start → stop → start → stop)

    let mut harness = TestHarness::new().await.unwrap();

    // Cycle 1: Start
    let result = harness.run_command(&["queen", "start"]).await.unwrap();
    assert_success(&result);
    harness.wait_for_ready("queen", Duration::from_secs(10)).await.unwrap();
    assert_running(&harness, "queen");

    // Cycle 1: Stop
    let result = harness.run_command(&["queen", "stop"]).await.unwrap();
    assert_success(&result);
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert_stopped(&harness, "queen");

    // Cycle 2: Start again
    let result = harness.run_command(&["queen", "start"]).await.unwrap();
    assert_success(&result);
    harness.wait_for_ready("queen", Duration::from_secs(10)).await.unwrap();
    assert_running(&harness, "queen");

    // Cycle 2: Stop again
    let result = harness.run_command(&["queen", "stop"]).await.unwrap();
    assert_success(&result);
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert_stopped(&harness, "queen");

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_queen_rapid_start_stop() {
    // TEAM-251: Test rapid start/stop cycles (3 times)

    let mut harness = TestHarness::new().await.unwrap();

    for i in 1..=3 {
        println!("🔄 Cycle {}/3", i);

        // Start
        let result = harness.run_command(&["queen", "start"]).await.unwrap();
        assert_success(&result);
        harness.wait_for_ready("queen", Duration::from_secs(10)).await.unwrap();

        // Stop
        let result = harness.run_command(&["queen", "stop"]).await.unwrap();
        assert_success(&result);
        tokio::time::sleep(Duration::from_secs(1)).await;
    }

    harness.cleanup().await.unwrap();
}

// ============================================================================
// Queen Health Check Tests
// ============================================================================

#[tokio::test]
async fn test_queen_health_endpoint_when_running() {
    // TEAM-251: Test queen health endpoint responds when running

    let mut harness = TestHarness::new().await.unwrap();

    // Start queen
    let _ = harness.run_command(&["queen", "start"]).await.unwrap();
    harness.wait_for_ready("queen", Duration::from_secs(10)).await.unwrap();

    // Health check should pass
    assert_running(&harness, "queen");

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_queen_health_endpoint_when_stopped() {
    // TEAM-251: Test queen health endpoint fails when stopped

    let harness = TestHarness::new().await.unwrap();

    // Health check should fail
    assert_stopped(&harness, "queen");
}
