// SSH communication tests
// Purpose: Test SSH operations between queen-rbee and rbee-hive
// Uses: RbeeSSHClient (russh library)
// Run with: cargo test --package xtask --test docker_ssh_tests --ignored

use std::time::Duration;
use xtask::integration::docker_harness::{DockerTestHarness, Topology};

#[tokio::test]
#[ignore]
async fn test_ssh_connection_to_hive() {
    let _harness = DockerTestHarness::new(Topology::Localhost)
        .await
        .expect("Failed to create Docker test harness");

    // Test: SSH connection using RbeeSSHClient
    // Port 2222 is mapped to container's port 22
    let mut client = queen_rbee_ssh_client::RbeeSSHClient::connect("localhost", 2222, "rbee")
        .await
        .expect("Failed to establish SSH connection to hive");

    // Execute simple command
    let (stdout, stderr, exit_code) =
        client.exec("echo 'test'").await.expect("Failed to execute command via SSH");

    assert_eq!(exit_code, 0, "Command should succeed");
    assert_eq!(stdout.trim(), "test");
    assert!(stderr.is_empty(), "No stderr expected");

    client.close().await.expect("Failed to close SSH connection");

    println!("✅ SSH connection test passed");
}

#[tokio::test]
#[ignore]
async fn test_ssh_authentication() {
    let _harness = DockerTestHarness::new(Topology::Localhost)
        .await
        .expect("Failed to create Docker test harness");

    // Test: SSH authentication
    let mut client = queen_rbee_ssh_client::RbeeSSHClient::connect("localhost", 2222, "rbee")
        .await
        .expect("SSH authentication should succeed with valid key");

    client.close().await.expect("Failed to close SSH connection");

    println!("✅ SSH authentication test passed");
}

#[tokio::test]
#[ignore]
async fn test_ssh_command_execution() {
    let _harness = DockerTestHarness::new(Topology::Localhost)
        .await
        .expect("Failed to create Docker test harness");

    let mut client = queen_rbee_ssh_client::RbeeSSHClient::connect("localhost", 2222, "rbee")
        .await
        .expect("Failed to establish SSH connection");

    // Test multiple commands
    let commands = vec![("whoami", "rbee"), ("pwd", "/home/rbee"), ("echo $HOME", "/home/rbee")];

    for (cmd, expected) in commands {
        let (stdout, _, exit_code) =
            client.exec(cmd).await.unwrap_or_else(|e| panic!("Command '{}' failed: {}", cmd, e));

        assert_eq!(exit_code, 0, "Command '{}' should succeed", cmd);
        assert!(
            stdout.trim().contains(expected),
            "Command '{}' expected '{}', got '{}'",
            cmd,
            expected,
            stdout.trim()
        );
    }

    client.close().await.expect("Failed to close SSH connection");

    println!("✅ SSH command execution test passed");
}

#[tokio::test]
#[ignore]
async fn test_ssh_binary_check() {
    let _harness = DockerTestHarness::new(Topology::Localhost)
        .await
        .expect("Failed to create Docker test harness");

    let mut client = queen_rbee_ssh_client::RbeeSSHClient::connect("localhost", 2222, "rbee")
        .await
        .expect("Failed to establish SSH connection");

    // Check if rbee-hive binary exists
    let (stdout, _, exit_code) = client
        .exec("ls -la ~/.local/bin/rbee-hive")
        .await
        .expect("Failed to check binary - did you run 'cargo build --bin rbee-hive' first?");

    assert_eq!(exit_code, 0, "Binary should exist");
    assert!(stdout.contains("rbee-hive"), "Binary should be named rbee-hive");
    assert!(stdout.contains("-rwx") || stdout.contains("x"), "Binary should be executable");

    // Check binary version
    let (stdout, _, exit_code) = client
        .exec("~/.local/bin/rbee-hive --version")
        .await
        .expect("Failed to get binary version");

    assert_eq!(exit_code, 0, "Version command should succeed");
    assert!(stdout.contains("rbee-hive"), "Version output should contain binary name");

    client.close().await.expect("Failed to close SSH connection");

    println!("✅ SSH binary check test passed");
}

#[tokio::test]
#[ignore]
async fn test_ssh_file_operations() {
    let _harness = DockerTestHarness::new(Topology::Localhost)
        .await
        .expect("Failed to create Docker test harness");

    let mut client = queen_rbee_ssh_client::RbeeSSHClient::connect("localhost", 2222, "rbee")
        .await
        .expect("Failed to establish SSH connection");

    // Create file
    let (_, _, exit_code) =
        client.exec("touch /tmp/ssh-test-file.txt").await.expect("Failed to create file");
    assert_eq!(exit_code, 0);

    // Write to file
    let (_, _, exit_code) = client
        .exec("echo 'SSH test content' > /tmp/ssh-test-file.txt")
        .await
        .expect("Failed to write to file");
    assert_eq!(exit_code, 0);

    // Read file
    let (stdout, _, exit_code) =
        client.exec("cat /tmp/ssh-test-file.txt").await.expect("Failed to read file");
    assert_eq!(exit_code, 0);
    assert_eq!(stdout.trim(), "SSH test content");

    // Delete file
    let (_, _, exit_code) =
        client.exec("rm /tmp/ssh-test-file.txt").await.expect("Failed to delete file");
    assert_eq!(exit_code, 0);

    client.close().await.expect("Failed to close SSH connection");

    println!("✅ SSH file operations test passed");
}

#[tokio::test]
#[ignore]
async fn test_ssh_concurrent_connections() {
    let _harness = DockerTestHarness::new(Topology::Localhost)
        .await
        .expect("Failed to create Docker test harness");

    // Test: Open 5 concurrent SSH connections
    let mut handles = vec![];

    for i in 0..5 {
        let handle = tokio::spawn(async move {
            let mut client =
                queen_rbee_ssh_client::RbeeSSHClient::connect("localhost", 2222, "rbee")
                    .await
                    .unwrap_or_else(|e| panic!("Connection {} failed: {}", i, e));

            let (stdout, _, exit_code) = client
                .exec(&format!("echo 'test-{}'", i))
                .await
                .unwrap_or_else(|e| panic!("Command {} failed: {}", i, e));

            assert_eq!(exit_code, 0);
            assert_eq!(stdout.trim(), format!("test-{}", i));

            client.close().await.expect("Failed to close connection");
            i
        });
        handles.push(handle);
    }

    // Wait for all connections to complete
    for handle in handles {
        handle.await.expect("Task panicked");
    }

    println!("✅ SSH concurrent connections test passed (5 connections)");
}

#[tokio::test]
#[ignore]
async fn test_ssh_connection_timeout() {
    let harness = DockerTestHarness::new(Topology::Localhost)
        .await
        .expect("Failed to create Docker test harness");

    // Establish connection first
    let mut client = queen_rbee_ssh_client::RbeeSSHClient::connect("localhost", 2222, "rbee")
        .await
        .expect("Failed to establish SSH connection");

    // Verify connection works
    let (_, _, exit_code) = client.exec("echo 'test'").await.expect("Command failed");
    assert_eq!(exit_code, 0);

    client.close().await.expect("Failed to close connection");

    // Kill hive to simulate SSH server down
    harness.kill("rbee-hive-localhost").await.expect("Failed to kill hive");

    // Wait for container to die
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Try to connect - should fail
    let result = queen_rbee_ssh_client::RbeeSSHClient::connect("localhost", 2222, "rbee").await;

    assert!(result.is_err(), "SSH connection should fail when server is down");

    println!("✅ SSH connection timeout test passed");
}

#[tokio::test]
#[ignore]
async fn test_ssh_environment_variables() {
    let _harness = DockerTestHarness::new(Topology::Localhost)
        .await
        .expect("Failed to create Docker test harness");

    let mut client = queen_rbee_ssh_client::RbeeSSHClient::connect("localhost", 2222, "rbee")
        .await
        .expect("Failed to establish SSH connection");

    // Test environment variables
    let env_vars = vec![("HOME", "/home/rbee"), ("USER", "rbee")];

    for (var, expected) in env_vars {
        let (stdout, _, exit_code) = client
            .exec(&format!("echo ${}", var))
            .await
            .unwrap_or_else(|e| panic!("Failed to get env var {}: {}", var, e));

        assert_eq!(exit_code, 0);
        assert_eq!(stdout.trim(), expected, "Environment variable {} should be {}", var, expected);
    }

    client.close().await.expect("Failed to close SSH connection");

    println!("✅ SSH environment variables test passed");
}
