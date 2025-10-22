// TEAM-210: Moved from lib.rs for better organization
// TEAM-220: Investigated - SSH connection testing documented

use anyhow::Result;
use observability_narration_core::Narration;
use queen_rbee_ssh_client::{test_ssh_connection, SshConfig};

// Actor and action constants
const ACTOR_HIVE_LIFECYCLE: &str = "🐝 hive-lifecycle";
const ACTION_SSH_TEST: &str = "ssh_test";

/// Request to test SSH connection
#[derive(Debug, Clone)]
pub struct SshTestRequest {
    /// SSH host address
    pub ssh_host: String,
    /// SSH port (default: 22)
    pub ssh_port: u16,
    /// SSH username
    pub ssh_user: String,
}

/// Response from SSH connection test
#[derive(Debug, Clone)]
pub struct SshTestResponse {
    /// Whether the connection was successful
    pub success: bool,
    /// Error message if connection failed
    pub error: Option<String>,
    /// Test command output (if successful)
    pub test_output: Option<String>,
}

/// Execute SSH connection test
///
/// TEAM-210: Moved from lib.rs
///
/// This function:
/// 1. Creates SSH client with provided credentials
/// 2. Attempts connection with timeout (5s)
/// 3. Runs simple test command (`echo test`)
/// 4. Returns success/failure with details
///
/// # Arguments
/// * `request` - SSH test request with host, port, and user
///
/// # Returns
/// * `Ok(SshTestResponse)` - Test result with success status
/// * `Err` - Critical error during test
pub async fn execute_ssh_test(request: SshTestRequest) -> Result<SshTestResponse> {
    let target = format!("{}@{}:{}", request.ssh_user, request.ssh_host, request.ssh_port);

    Narration::new(ACTOR_HIVE_LIFECYCLE, ACTION_SSH_TEST, &target)
        .human(format!("🔐 Testing SSH connection to {}", target))
        .emit();

    // Create SSH config from request
    let config = SshConfig {
        host: request.ssh_host,
        port: request.ssh_port,
        user: request.ssh_user,
        timeout_secs: 5,
    };

    // Test SSH connection using ssh-client crate
    let result = test_ssh_connection(config).await?;

    // Convert SshTestResult to SshTestResponse
    let response = SshTestResponse {
        success: result.success,
        error: result.error,
        test_output: result.test_output,
    };

    if response.success {
        Narration::new(ACTOR_HIVE_LIFECYCLE, ACTION_SSH_TEST, "success")
            .human(format!("✅ SSH test successful: {}", target))
            .emit();
    } else {
        Narration::new(ACTOR_HIVE_LIFECYCLE, ACTION_SSH_TEST, "failed")
            .human(format!(
                "❌ SSH test failed: {}",
                response.error.as_deref().unwrap_or("unknown")
            ))
            .emit();
    }

    Ok(response)
}
