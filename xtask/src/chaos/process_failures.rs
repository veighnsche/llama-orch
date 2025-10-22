// TEAM-252: Process failure tests
// Purpose: Test behavior when processes crash


#[tokio::test]
async fn test_queen_crash_during_operation() {
    // TEAM-252: Test queen crash mid-operation

    let mut harness = TestHarness::new().await.unwrap();

    // Start queen
    let _ = harness.run_command(&["queen", "start"]).await.unwrap();
    harness.wait_for_ready("queen", Duration::from_secs(10)).await.unwrap();

    // Kill queen forcefully
    harness.kill_process("queen-rbee").await.unwrap();

    // Try to run command
    let result = harness.run_command(&["hive", "list"]).await.unwrap();

    // Should detect queen is down
    assert_failure(&result);
    assert_output_contains(&result, "queen");

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_hive_crash_during_operation() {
    // TEAM-252: Test hive crash mid-operation

    let mut harness = TestHarness::new().await.unwrap();

    // Start both
    harness.run_command(&["hive", "start"]).await.unwrap();
    harness.wait_for_ready("hive", Duration::from_secs(10)).await.unwrap();

    // Kill hive forcefully
    harness.kill_process("rbee-hive").await.unwrap();

    // Try to run command
    let result = harness.run_command(&["hive", "status"]).await.unwrap();

    // Should detect hive is down
    assert_failure(&result);

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_process_hang() {
    // TEAM-252: Test process hang (timeout)

    let mut harness = TestHarness::new().await.unwrap();

    // Start both
    harness.run_command(&["hive", "start"]).await.unwrap();
    harness.wait_for_ready("hive", Duration::from_secs(10)).await.unwrap();

    // Run command with timeout measurement
    let start = std::time::Instant::now();
    let result = harness.run_command(&["hive", "list"]).await.unwrap();
    let elapsed = start.elapsed();

    // Should complete (not hang)
    assert_success(&result);
    assert!(elapsed < Duration::from_secs(30), "Command hung: {:?}", elapsed);

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_queen_restart_recovery() {
    // TEAM-252: Test recovery after queen restart

    let mut harness = TestHarness::new().await.unwrap();

    // Start queen
    harness.run_command(&["queen", "start"]).await.unwrap();
    harness.wait_for_ready("queen", Duration::from_secs(10)).await.unwrap();

    // Verify it works
    let result = harness.run_command(&["hive", "list"]).await.unwrap();
    assert_success(&result);

    // Kill queen
    harness.kill_process("queen-rbee").await.unwrap();

    // Wait for it to die
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Restart queen
    harness.run_command(&["queen", "start"]).await.unwrap();
    harness.wait_for_ready("queen", Duration::from_secs(10)).await.unwrap();

    // Should work again
    let result = harness.run_command(&["hive", "list"]).await.unwrap();
    assert_success(&result);

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_hive_restart_recovery() {
    // TEAM-252: Test recovery after hive restart

    let mut harness = TestHarness::new().await.unwrap();

    // Start both
    harness.run_command(&["hive", "start"]).await.unwrap();
    harness.wait_for_ready("hive", Duration::from_secs(10)).await.unwrap();

    // Verify hive is running
    let result = harness.run_command(&["hive", "status"]).await.unwrap();
    assert_success(&result);

    // Kill hive
    harness.kill_process("rbee-hive").await.unwrap();

    // Wait for it to die
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Restart hive
    harness.run_command(&["hive", "start"]).await.unwrap();
    harness.wait_for_ready("hive", Duration::from_secs(10)).await.unwrap();

    // Should work again
    let result = harness.run_command(&["hive", "status"]).await.unwrap();
    assert_success(&result);

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_rapid_process_restart() {
    // TEAM-252: Test rapid restart of processes

    let mut harness = TestHarness::new().await.unwrap();

    // Start and stop queen multiple times
    for i in 0..3 {
        println!("Iteration {}", i + 1);

        harness.run_command(&["queen", "start"]).await.unwrap();
        harness.wait_for_ready("queen", Duration::from_secs(10)).await.unwrap();

        let result = harness.run_command(&["hive", "list"]).await.unwrap();
        assert_success(&result);

        harness.run_command(&["queen", "stop"]).await.unwrap();
        tokio::time::sleep(Duration::from_secs(1)).await;
    }

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_process_state_consistency() {
    // TEAM-252: Test process state remains consistent after crash

    let mut harness = TestHarness::new().await.unwrap();

    // Start both
    harness.run_command(&["hive", "start"]).await.unwrap();
    harness.wait_for_ready("hive", Duration::from_secs(10)).await.unwrap();

    // Get initial state
    let initial_state = harness.get_state().await;
    assert_eq!(initial_state.queen, crate::integration::harness::ProcessState::Running);
    assert_eq!(initial_state.hive, crate::integration::harness::ProcessState::Running);

    // Kill hive
    harness.kill_process("rbee-hive").await.unwrap();

    // Check state
    tokio::time::sleep(Duration::from_secs(1)).await;
    let state_after_crash = harness.get_state().await;
    assert_eq!(state_after_crash.queen, crate::integration::harness::ProcessState::Running);
    assert_eq!(state_after_crash.hive, crate::integration::harness::ProcessState::Stopped);

    harness.cleanup().await.unwrap();
}
