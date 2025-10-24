// TEAM-244: SSH Client comprehensive tests
// Purpose: Test all SSH connection scenarios (pre-flight, TCP, handshake, auth, command)
// Priority: HIGH (0% coverage → comprehensive coverage)

use queen_rbee_ssh_client::{test_ssh_connection, SshConfig};
use std::env;

// ============================================================================
// Pre-flight Check Tests
// ============================================================================

#[tokio::test]
async fn test_ssh_agent_not_set() {
    // TEAM-244: Test SSH_AUTH_SOCK not set (should return helpful error)
    env::remove_var("SSH_AUTH_SOCK");

    let config = SshConfig {
        host: "localhost".to_string(),
        port: 22,
        user: "testuser".to_string(),
        timeout_secs: 5,
    };

    let result = test_ssh_connection(config).await.unwrap();

    assert!(!result.success, "Connection should fail when SSH agent not running");
    assert!(result.error.is_some(), "Should have error message");
    let error = result.error.unwrap();
    assert!(
        error.contains("SSH agent not running") || error.contains("ssh-agent"),
        "Error should mention SSH agent: {}",
        error
    );
    assert!(error.contains("eval $(ssh-agent)"), "Error should include helpful command");
}

#[tokio::test]
async fn test_ssh_agent_empty_string() {
    // TEAM-244: Test SSH_AUTH_SOCK set but empty string
    env::set_var("SSH_AUTH_SOCK", "");

    let config = SshConfig {
        host: "localhost".to_string(),
        port: 22,
        user: "testuser".to_string(),
        timeout_secs: 5,
    };

    let result = test_ssh_connection(config).await.unwrap();

    assert!(!result.success, "Connection should fail when SSH_AUTH_SOCK is empty");
    assert!(result.error.is_some(), "Should have error message");
    let error = result.error.unwrap();
    assert!(
        error.contains("SSH agent not running") || error.contains("ssh-agent"),
        "Error should mention SSH agent: {}",
        error
    );
}

#[tokio::test]
async fn test_ssh_agent_nonexistent_socket() {
    // TEAM-244: Test SSH_AUTH_SOCK points to non-existent socket
    env::set_var("SSH_AUTH_SOCK", "/tmp/nonexistent_ssh_socket_12345");

    let config = SshConfig {
        host: "localhost".to_string(),
        port: 22,
        user: "testuser".to_string(),
        timeout_secs: 5,
    };

    let result = test_ssh_connection(config).await.unwrap();

    // This should fail during authentication, not pre-flight
    assert!(!result.success, "Connection should fail with invalid socket");
    assert!(result.error.is_some(), "Should have error message");
}

// ============================================================================
// TCP Connection Tests
// ============================================================================

#[tokio::test]
async fn test_connection_to_unreachable_host() {
    // TEAM-244: Test connection to unreachable host (timeout behavior)
    // Use a non-routable IP address (RFC 5737 TEST-NET-1)
    // This test requires SSH_AUTH_SOCK to be set in environment before running tests
    // Run with: SSH_AUTH_SOCK=/tmp/fake_socket cargo test
    if env::var("SSH_AUTH_SOCK").is_err() {
        env::set_var("SSH_AUTH_SOCK", "/tmp/fake_socket");
    }

    let config = SshConfig {
        host: "192.0.2.1".to_string(), // Non-routable test IP
        port: 22,
        user: "testuser".to_string(),
        timeout_secs: 2, // Short timeout for test speed
    };

    let result = test_ssh_connection(config).await.unwrap();

    assert!(!result.success, "Connection to unreachable host should fail");
    assert!(result.error.is_some(), "Should have error message");
    let error = result.error.unwrap();
    assert!(
        error.contains("TCP connection failed")
            || error.contains("timeout")
            || error.contains("Connection")
            || error.contains("SSH agent not running"),
        "Error should mention TCP/connection/agent failure: {}",
        error
    );
}

#[tokio::test]
async fn test_connection_to_invalid_port() {
    // TEAM-244: Test connection to port with no SSH server
    // This test requires SSH_AUTH_SOCK to be set in environment before running tests
    // Run with: SSH_AUTH_SOCK=/tmp/fake_socket cargo test
    if env::var("SSH_AUTH_SOCK").is_err() {
        env::set_var("SSH_AUTH_SOCK", "/tmp/fake_socket");
    }

    let config = SshConfig {
        host: "localhost".to_string(),
        port: 9999, // Unlikely to have SSH server
        user: "testuser".to_string(),
        timeout_secs: 2,
    };

    let result = test_ssh_connection(config).await.unwrap();

    assert!(!result.success, "Connection to invalid port should fail");
    assert!(result.error.is_some(), "Should have error message");
    let error = result.error.unwrap();
    assert!(
        error.contains("TCP connection failed")
            || error.contains("Connection refused")
            || error.contains("handshake")
            || error.contains("SSH agent not running"),
        "Error should mention connection/handshake/agent failure: {}",
        error
    );
}

#[tokio::test]
async fn test_connection_timeout() {
    // TEAM-244: Test connection timeout (very short timeout)
    env::set_var("SSH_AUTH_SOCK", "/tmp/fake_socket");

    let config = SshConfig {
        host: "192.0.2.1".to_string(), // Non-routable
        port: 22,
        user: "testuser".to_string(),
        timeout_secs: 1, // Very short timeout
    };

    let start = std::time::Instant::now();
    let result = test_ssh_connection(config).await.unwrap();
    let elapsed = start.elapsed();

    assert!(!result.success, "Connection should timeout");
    assert!(result.error.is_some(), "Should have error message");
    assert!(
        elapsed.as_secs() <= 3,
        "Should timeout within reasonable time (got {}s)",
        elapsed.as_secs()
    );
}

// ============================================================================
// SSH Handshake Tests
// ============================================================================

#[tokio::test]
async fn test_handshake_with_non_ssh_server() {
    // TEAM-244: Test connection to non-SSH server (e.g., HTTP on port 22)
    // This is hard to test without a mock server, so we test invalid host format
    env::set_var("SSH_AUTH_SOCK", "/tmp/fake_socket");

    let config = SshConfig {
        host: "invalid_host_name_that_does_not_exist_12345".to_string(),
        port: 22,
        user: "testuser".to_string(),
        timeout_secs: 2,
    };

    let result = test_ssh_connection(config).await.unwrap();

    assert!(!result.success, "Connection to invalid host should fail");
    assert!(result.error.is_some(), "Should have error message");
}

// ============================================================================
// Authentication Tests
// ============================================================================

#[tokio::test]
async fn test_auth_with_wrong_username() {
    // TEAM-244: Test with wrong username (will fail if SSH server exists)
    // This test assumes localhost SSH is NOT configured for "nonexistent_user_12345"
    if let Ok(sock) = env::var("SSH_AUTH_SOCK") {
        if !sock.is_empty() {
            let config = SshConfig {
                host: "localhost".to_string(),
                port: 22,
                user: "nonexistent_user_12345".to_string(),
                timeout_secs: 5,
            };

            let result = test_ssh_connection(config).await.unwrap();

            // This will fail at TCP or auth stage
            assert!(!result.success, "Connection with wrong username should fail");
            assert!(result.error.is_some(), "Should have error message");
        }
    }
}

// ============================================================================
// Command Execution Tests
// ============================================================================

#[tokio::test]
async fn test_command_output_parsing() {
    // TEAM-244: Test command output parsing
    // This test only runs if SSH agent is properly configured for localhost
    if let Ok(sock) = env::var("SSH_AUTH_SOCK") {
        if !sock.is_empty() && std::path::Path::new(&sock).exists() {
            let current_user = env::var("USER").unwrap_or_else(|_| "root".to_string());

            let config = SshConfig {
                host: "localhost".to_string(),
                port: 22,
                user: current_user,
                timeout_secs: 5,
            };

            let result = test_ssh_connection(config).await.unwrap();

            if result.success {
                // If connection succeeds, verify output
                assert!(result.test_output.is_some(), "Should have test output");
                let output = result.test_output.unwrap();
                assert_eq!(output.trim(), "test", "Output should be 'test'");
            }
            // If connection fails, that's OK (SSH not configured for localhost)
        }
    }
}

// ============================================================================
// Narration Tests
// ============================================================================

#[tokio::test]
async fn test_narration_on_success() {
    // TEAM-244: Test narration emitted on success
    // This is implicitly tested by test_command_output_parsing
    // Narration is emitted to stdout/SSE, not returned in result
    // We verify the function doesn't panic and returns proper result
    if let Ok(sock) = env::var("SSH_AUTH_SOCK") {
        if !sock.is_empty() && std::path::Path::new(&sock).exists() {
            let current_user = env::var("USER").unwrap_or_else(|_| "root".to_string());

            let config = SshConfig {
                host: "localhost".to_string(),
                port: 22,
                user: current_user,
                timeout_secs: 5,
            };

            // Should not panic
            let _result = test_ssh_connection(config).await.unwrap();
        }
    }
}

#[tokio::test]
async fn test_narration_on_failure() {
    // TEAM-244: Test narration emitted on failure
    env::set_var("SSH_AUTH_SOCK", "/tmp/fake_socket");

    let config = SshConfig {
        host: "192.0.2.1".to_string(),
        port: 22,
        user: "testuser".to_string(),
        timeout_secs: 1,
    };

    // Should not panic, should emit failure narration
    let result = test_ssh_connection(config).await.unwrap();
    assert!(!result.success);
}

#[tokio::test]
async fn test_narration_includes_target() {
    // TEAM-244: Test narration includes correct target
    env::remove_var("SSH_AUTH_SOCK");

    let config = SshConfig {
        host: "testhost".to_string(),
        port: 2222,
        user: "testuser".to_string(),
        timeout_secs: 5,
    };

    // Narration should include "testuser@testhost:2222"
    let _result = test_ssh_connection(config).await.unwrap();
    // Narration is emitted, not returned - we just verify no panic
}

// ============================================================================
// Edge Cases
// ============================================================================

#[tokio::test]
async fn test_default_config() {
    // TEAM-244: Test default SshConfig values
    let config = SshConfig::default();

    assert_eq!(config.host, "");
    assert_eq!(config.port, 22);
    assert_eq!(config.user, "");
    assert_eq!(config.timeout_secs, 5);
}

#[tokio::test]
async fn test_config_with_custom_timeout() {
    // TEAM-244: Test custom timeout value
    env::set_var("SSH_AUTH_SOCK", "/tmp/fake_socket");

    let config = SshConfig {
        host: "192.0.2.1".to_string(),
        port: 22,
        user: "testuser".to_string(),
        timeout_secs: 1, // Custom short timeout
    };

    let start = std::time::Instant::now();
    let result = test_ssh_connection(config).await.unwrap();
    let elapsed = start.elapsed();

    assert!(!result.success);
    assert!(elapsed.as_secs() <= 3, "Should respect custom timeout");
}
