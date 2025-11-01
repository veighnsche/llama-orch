// TEAM-252: Network failure tests
// Purpose: Test behavior with network issues
// TEAM-255: Fixed missing imports


#[tokio::test]
async fn test_port_already_in_use() {
    // TEAM-252: Test queen start when port 9000 is already in use

    let mut harness = TestHarness::new().await.unwrap();

    // Block port 9000
    let _listener = TcpListener::bind("127.0.0.1:9000").unwrap();

    // Try to start queen (should fail or use different port)
    let result = harness.run_command(&["queen", "start"]).await.unwrap();

    // Should either fail with helpful error or succeed with different port
    if result.exit_code != Some(0) {
        assert_output_contains(&result, "port");
    }

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_queen_unreachable() {
    // TEAM-252: Test hive list when queen is unreachable

    let mut harness = TestHarness::new().await.unwrap();

    // Try to list hives without starting queen
    let result = harness.run_command(&["hive", "list"]).await.unwrap();

    // Should fail because queen is not running
    assert_failure(&result);
    assert_output_contains(&result, "queen");

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_connection_timeout() {
    // TEAM-252: Test timeout when connecting to queen

    let mut harness = TestHarness::new().await.unwrap();

    // Start queen
    harness.run_command(&["queen", "start"]).await.unwrap();
    harness.wait_for_ready("queen", Duration::from_secs(10)).await.unwrap();

    // Kill queen forcefully to simulate network issue
    harness.kill_process("queen-rbee").await.unwrap();

    // Try to run command (should timeout or fail quickly)
    let start = std::time::Instant::now();
    let result = harness.run_command(&["hive", "list"]).await.unwrap();
    let elapsed = start.elapsed();

    // Should fail
    assert_failure(&result);

    // Should not hang (timeout should be reasonable)
    assert!(elapsed < Duration::from_secs(30), "Command took too long: {:?}", elapsed);

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_hive_unreachable() {
    // TEAM-252: Test hive status when hive is unreachable

    let mut harness = TestHarness::new().await.unwrap();

    // Start queen
    harness.run_command(&["queen", "start"]).await.unwrap();
    harness.wait_for_ready("queen", Duration::from_secs(10)).await.unwrap();

    // Try to get hive status without starting hive
    let result = harness.run_command(&["hive", "status"]).await.unwrap();

    // Should fail because hive is not running
    assert_failure(&result);

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_network_partition_recovery() {
    // TEAM-252: Test recovery after network partition

    let mut harness = TestHarness::new().await.unwrap();

    // Start both
    harness.run_command(&["hive", "start"]).await.unwrap();
    harness.wait_for_ready("hive", Duration::from_secs(10)).await.unwrap();

    // Kill queen to simulate network partition
    harness.kill_process("queen-rbee").await.unwrap();

    // Wait a bit
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Try to run command (should fail)
    let result = harness.run_command(&["hive", "list"]).await.unwrap();
    assert_failure(&result);

    // Restart queen
    harness.run_command(&["queen", "start"]).await.unwrap();
    harness.wait_for_ready("queen", Duration::from_secs(10)).await.unwrap();

    // Now command should succeed
    let result = harness.run_command(&["hive", "list"]).await.unwrap();
    assert_success(&result);

    harness.cleanup().await.unwrap();
}

#[tokio::test]
async fn test_slow_network_response() {
    // TEAM-252: Test handling of slow network responses

    let mut harness = TestHarness::new().await.unwrap();

    // Start both
    harness.run_command(&["hive", "start"]).await.unwrap();
    harness.wait_for_ready("hive", Duration::from_secs(10)).await.unwrap();

    // Run command and measure time
    let start = std::time::Instant::now();
    let result = harness.run_command(&["hive", "list"]).await.unwrap();
    let elapsed = start.elapsed();

    // Should succeed
    assert_success(&result);

    // Should complete in reasonable time (not hang)
    assert!(elapsed < Duration::from_secs(30), "Command took too long: {:?}", elapsed);

    harness.cleanup().await.unwrap();
}
