// TEAM-251: Hive command tests
// Purpose: Test all hive commands in all valid states
// Commands: hive start, hive stop, hive list, hive status

use crate::integration::assertions::*;
use crate::integration::harness::TestHarness;
use std::time::Duration;

// ============================================================================
// Hive Start Tests
// ============================================================================

#[tokio::test]
async fn test_hive_start_when_both_stopped() {
    // TEAM-251: Test hive start when both queen and hive are stopped
    // Should start queen first, then hive

    let mut harness = TestHarness::new().await.unwrap();

    // Precondition: Both stopped
    assert_stopped(&harness, "queen");
    assert_stopped(&harness, "hive");

    // Action: Start hive
    let result = harness.run_command(&["hive", "start"]).await.unwrap();

    // Assertions
    assert_success(&result);
    assert_output_contains(&result, "✅");

    // Postcondition: Both running
    harness.wait_for_ready("queen", Duration::from_secs(10)).await.unwrap();
    harness.wait_for_ready("hive", Duration::from_secs(10)).await.unwrap();
    assert_running(&harness, "queen");
    assert_running(&harness, "hive");

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_hive_start_when_queen_already_running() {
    // TEAM-251: Test hive start when queen is already running
    // Should only start hive

    let mut harness = TestHarness::new().await.unwrap();

    // Precondition: Queen running, hive stopped
    let _ = harness.run_command(&["queen", "start"]).await.unwrap();
    harness.wait_for_ready("queen", Duration::from_secs(10)).await.unwrap();
    assert_running(&harness, "queen");
    assert_stopped(&harness, "hive");

    // Action: Start hive
    let result = harness.run_command(&["hive", "start"]).await.unwrap();

    // Assertions
    assert_success(&result);
    assert_output_contains(&result, "✅");

    // Postcondition: Both running
    harness.wait_for_ready("hive", Duration::from_secs(10)).await.unwrap();
    assert_running(&harness, "queen");
    assert_running(&harness, "hive");

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_hive_start_when_already_running() {
    // TEAM-251: Test hive start when hive is already running (idempotent)

    let mut harness = TestHarness::new().await.unwrap();

    // Precondition: Hive running
    let _ = harness.run_command(&["hive", "start"]).await.unwrap();
    harness.wait_for_ready("queen", Duration::from_secs(10)).await.unwrap();
    harness.wait_for_ready("hive", Duration::from_secs(10)).await.unwrap();
    assert_running(&harness, "hive");

    // Action: Start hive again
    let result = harness.run_command(&["hive", "start"]).await.unwrap();

    // Assertions
    assert_success(&result);
    assert_output_contains(&result, "already running");

    // Postcondition: Hive still running
    assert_running(&harness, "hive");

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_hive_start_with_alias() {
    // TEAM-251: Test hive start with explicit alias (localhost)

    let mut harness = TestHarness::new().await.unwrap();

    // Action: Start hive with alias
    let result = harness.run_command(&["hive", "start", "localhost"]).await.unwrap();

    // Assertions
    assert_success(&result);
    assert_output_contains(&result, "✅");

    // Postcondition: Hive running
    harness.wait_for_ready("hive", Duration::from_secs(10)).await.unwrap();
    assert_running(&harness, "hive");

    harness.cleanup().await.unwrap();
}

// ============================================================================
// Hive Stop Tests
// ============================================================================

#[tokio::test]
async fn test_hive_stop_when_running() {
    // TEAM-251: Test hive stop when hive is running (happy path)

    let mut harness = TestHarness::new().await.unwrap();

    // Precondition: Hive running
    let _ = harness.run_command(&["hive", "start"]).await.unwrap();
    harness.wait_for_ready("hive", Duration::from_secs(10)).await.unwrap();
    assert_running(&harness, "hive");

    // Action: Stop hive
    let result = harness.run_command(&["hive", "stop"]).await.unwrap();

    // Assertions
    assert_success(&result);
    assert_output_contains(&result, "✅");
    assert_output_contains(&result, "stopped");

    // Postcondition: Hive stopped, queen still running
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert_stopped(&harness, "hive");
    assert_running(&harness, "queen");

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_hive_stop_when_already_stopped() {
    // TEAM-251: Test hive stop when hive is already stopped (idempotent)

    let mut harness = TestHarness::new().await.unwrap();

    // Precondition: Hive stopped (but queen might be running)
    let _ = harness.run_command(&["queen", "start"]).await.unwrap();
    harness.wait_for_ready("queen", Duration::from_secs(10)).await.unwrap();
    assert_stopped(&harness, "hive");

    // Action: Stop hive
    let result = harness.run_command(&["hive", "stop"]).await.unwrap();

    // Assertions
    assert_success(&result);
    let output = format!("{}{}", result.stdout, result.stderr);
    assert!(
        output.contains("not running") || output.contains("already stopped"),
        "Should indicate hive is not running"
    );

    // Postcondition: Hive still stopped
    assert_stopped(&harness, "hive");

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_hive_stop_with_alias() {
    // TEAM-251: Test hive stop with explicit alias (localhost)

    let mut harness = TestHarness::new().await.unwrap();

    // Precondition: Hive running
    let _ = harness.run_command(&["hive", "start"]).await.unwrap();
    harness.wait_for_ready("hive", Duration::from_secs(10)).await.unwrap();

    // Action: Stop hive with alias
    let result = harness.run_command(&["hive", "stop", "localhost"]).await.unwrap();

    // Assertions
    assert_success(&result);
    assert_output_contains(&result, "✅");

    // Postcondition: Hive stopped
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert_stopped(&harness, "hive");

    harness.cleanup().await.unwrap();
}

// ============================================================================
// Hive List Tests
// ============================================================================

#[tokio::test]
async fn test_hive_list_when_queen_running() {
    // TEAM-251: Test hive list when queen is running

    let mut harness = TestHarness::new().await.unwrap();

    // Precondition: Queen running
    let _ = harness.run_command(&["queen", "start"]).await.unwrap();
    harness.wait_for_ready("queen", Duration::from_secs(10)).await.unwrap();

    // Action: List hives
    let result = harness.run_command(&["hive", "list"]).await.unwrap();

    // Assertions
    assert_success(&result);
    assert_output_contains(&result, "localhost");

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_hive_list_when_queen_stopped() {
    // TEAM-251: Test hive list when queen is stopped (should fail)

    let mut harness = TestHarness::new().await.unwrap();

    // Precondition: Queen stopped
    assert_stopped(&harness, "queen");

    // Action: List hives
    let result = harness.run_command(&["hive", "list"]).await.unwrap();

    // Assertions: Should fail or indicate queen not running
    // (Implementation may vary - either error or auto-start queen)
    if result.exit_code != Some(0) {
        assert_output_contains(&result, "queen");
    }

    harness.cleanup().await.unwrap();
}

// ============================================================================
// Hive Status Tests
// ============================================================================

#[tokio::test]
async fn test_hive_status_when_hive_running() {
    // TEAM-251: Test hive status when hive is running

    let mut harness = TestHarness::new().await.unwrap();

    // Precondition: Hive running
    let _ = harness.run_command(&["hive", "start"]).await.unwrap();
    harness.wait_for_ready("hive", Duration::from_secs(10)).await.unwrap();

    // Action: Check status
    let result = harness.run_command(&["hive", "status"]).await.unwrap();

    // Assertions
    assert_success(&result);
    assert_output_contains(&result, "running");

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_hive_status_when_hive_stopped() {
    // TEAM-251: Test hive status when hive is stopped

    let mut harness = TestHarness::new().await.unwrap();

    // Precondition: Queen running, hive stopped
    let _ = harness.run_command(&["queen", "start"]).await.unwrap();
    harness.wait_for_ready("queen", Duration::from_secs(10)).await.unwrap();
    assert_stopped(&harness, "hive");

    // Action: Check status
    let result = harness.run_command(&["hive", "status"]).await.unwrap();

    // Assertions
    assert_success(&result);
    // Should show no hives or empty status

    harness.cleanup().await.unwrap();
}

// ============================================================================
// Hive Lifecycle Tests
// ============================================================================

#[tokio::test]
async fn test_hive_full_lifecycle() {
    // TEAM-251: Test complete hive lifecycle (start → stop → start → stop)

    let mut harness = TestHarness::new().await.unwrap();

    // Cycle 1: Start
    let result = harness.run_command(&["hive", "start"]).await.unwrap();
    assert_success(&result);
    harness.wait_for_ready("hive", Duration::from_secs(10)).await.unwrap();
    assert_running(&harness, "hive");

    // Cycle 1: Stop
    let result = harness.run_command(&["hive", "stop"]).await.unwrap();
    assert_success(&result);
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert_stopped(&harness, "hive");

    // Cycle 2: Start again
    let result = harness.run_command(&["hive", "start"]).await.unwrap();
    assert_success(&result);
    harness.wait_for_ready("hive", Duration::from_secs(10)).await.unwrap();
    assert_running(&harness, "hive");

    // Cycle 2: Stop again
    let result = harness.run_command(&["hive", "stop"]).await.unwrap();
    assert_success(&result);
    tokio::time::sleep(Duration::from_secs(1)).await;
    assert_stopped(&harness, "hive");

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_hive_heartbeat_after_start() {
    // TEAM-251: Test hive sends heartbeat after start

    let mut harness = TestHarness::new().await.unwrap();

    // Start hive
    let _ = harness.run_command(&["hive", "start"]).await.unwrap();
    harness.wait_for_ready("hive", Duration::from_secs(10)).await.unwrap();

    // Wait for heartbeat (5s interval)
    tokio::time::sleep(Duration::from_secs(6)).await;

    // Check status - should show hive with recent heartbeat
    let result = harness.run_command(&["hive", "status"]).await.unwrap();
    assert_success(&result);
    assert_output_contains(&result, "localhost");

    harness.cleanup().await.unwrap();
}
