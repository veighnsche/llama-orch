// Error verification helpers
// Created by: TEAM-062
//
// ⚠️ ⚠️ ⚠️ CRITICAL WARNING - DO NOT REMOVE THESE WARNINGS ⚠️ ⚠️ ⚠️
// ⚠️ CRITICAL: BDD tests MUST connect to product code from /bin/
// ⚠️ This is normal BDD behavior - connect to rbee-hive and llm-worker-rbee
// ⚠️ See TEAM_063_REAL_HANDOFF.md
// ⚠️ DEVELOPERS: You are NOT ALLOWED to remove these warnings!
// ⚠️ ⚠️ ⚠️ END CRITICAL WARNING ⚠️ ⚠️ ⚠️
//
// Provides reusable functions for verifying error conditions in BDD tests
// Modified by: TEAM-064 (added explicit warning preservation notice)

use crate::steps::world::{ErrorResponse, World};
use anyhow::{Context, Result};

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Error Verification
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Verify that an error occurred
pub fn verify_error_occurred(world: &World) -> Result<()> {
    if world.last_error.is_none() {
        anyhow::bail!("Expected error but none occurred");
    }
    Ok(())
}

/// Verify error message contains expected text
pub fn verify_error_message_contains(world: &World, expected: &str) -> Result<()> {
    let error = world.last_error.as_ref().context("No error occurred")?;

    if !error.message.contains(expected) {
        anyhow::bail!("Error message '{}' does not contain '{}'", error.message, expected);
    }
    Ok(())
}

/// Verify error code matches expected
pub fn verify_error_code(world: &World, expected_code: &str) -> Result<()> {
    let error = world.last_error.as_ref().context("No error occurred")?;

    if error.code != expected_code {
        anyhow::bail!("Expected error code '{}' but got '{}'", expected_code, error.code);
    }
    Ok(())
}

/// Verify exit code matches expected
pub fn verify_exit_code(world: &World, expected: i32) -> Result<()> {
    let actual = world.last_exit_code.context("No exit code recorded")?;

    if actual != expected {
        anyhow::bail!("Expected exit code {} but got {}", expected, actual);
    }
    Ok(())
}

/// Verify HTTP status code matches expected
pub fn verify_http_status(world: &World, expected: u16) -> Result<()> {
    let actual = world.last_http_status.context("No HTTP status recorded")?;

    if actual != expected {
        anyhow::bail!("Expected HTTP status {} but got {}", expected, actual);
    }
    Ok(())
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Error Response Parsing
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Parse error response from HTTP body
///
/// Expected format:
/// ```json
/// {
///   "error": {
///     "code": "ERROR_CODE",
///     "message": "Human-readable message",
///     "details": { ... }
///   }
/// }
/// ```
pub fn parse_error_response(body: &str) -> Result<ErrorResponse> {
    let json: serde_json::Value =
        serde_json::from_str(body).context("Failed to parse error response as JSON")?;

    let error_obj = json.get("error").context("Response missing 'error' field")?;

    Ok(ErrorResponse {
        code: error_obj.get("code").and_then(|v| v.as_str()).unwrap_or("UNKNOWN").to_string(),
        message: error_obj.get("message").and_then(|v| v.as_str()).unwrap_or("").to_string(),
        details: error_obj.get("details").cloned(),
    })
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Resource Checks
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Check available RAM in MB
#[cfg(target_os = "linux")]
pub fn check_available_ram_mb() -> Result<usize> {
    let meminfo = std::fs::read_to_string("/proc/meminfo")?;
    let available_line = meminfo
        .lines()
        .find(|line| line.starts_with("MemAvailable:"))
        .context("MemAvailable not found in /proc/meminfo")?;

    let kb: usize = available_line
        .split_whitespace()
        .nth(1)
        .context("Failed to parse MemAvailable")?
        .parse()?;

    Ok(kb / 1024) // Convert KB to MB
}

#[cfg(not(target_os = "linux"))]
pub fn check_available_ram_mb() -> Result<usize> {
    // Fallback for non-Linux systems
    Ok(8000)
}

/// Check if port is available
pub async fn is_port_available(port: u16) -> bool {
    tokio::net::TcpListener::bind(format!("127.0.0.1:{}", port)).await.is_ok()
}

/// Find next available port starting from base
pub async fn find_available_port(base: u16) -> Result<u16> {
    for port in base..base + 100 {
        if is_port_available(port).await {
            return Ok(port);
        }
    }
    anyhow::bail!("No available ports in range {}-{}", base, base + 100)
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Process Management
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Check if process is running
pub fn is_process_running(pid: u32) -> bool {
    #[cfg(unix)]
    {
        use nix::sys::signal::{kill, Signal};
        use nix::unistd::Pid;
        // Send signal 0 to check if process exists (doesn't actually send signal)
        kill(Pid::from_raw(pid as i32), None).is_ok()
    }

    #[cfg(not(unix))]
    {
        // Fallback: assume running
        true
    }
}

/// Wait for process to exit with timeout
pub async fn wait_for_process_exit(pid: u32, timeout: std::time::Duration) -> Result<()> {
    let start = std::time::Instant::now();

    loop {
        if !is_process_running(pid) {
            return Ok(());
        }

        if start.elapsed() > timeout {
            anyhow::bail!("Process {} did not exit within {:?}", pid, timeout);
        }

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Retry Logic
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Retry operation with exponential backoff
pub async fn retry_with_backoff<F, Fut, T>(
    operation: F,
    max_attempts: u32,
    operation_name: &str,
) -> Result<T>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = Result<T>>,
{
    let mut attempt = 1;
    let mut delay = std::time::Duration::from_millis(100);

    loop {
        match operation().await {
            Ok(result) => {
                if attempt > 1 {
                    tracing::info!("✅ {} succeeded on attempt {}", operation_name, attempt);
                }
                return Ok(result);
            }
            Err(e) if attempt >= max_attempts => {
                tracing::error!(
                    "❌ {} failed after {} attempts: {}",
                    operation_name,
                    max_attempts,
                    e
                );
                return Err(e);
            }
            Err(e) => {
                tracing::warn!(
                    "⚠️  {} failed (attempt {}/{}): {}",
                    operation_name,
                    attempt,
                    max_attempts,
                    e
                );
                tokio::time::sleep(delay).await;
                delay *= 2; // Exponential backoff
                attempt += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_error_response() {
        let json = r#"{"error":{"code":"TEST_ERROR","message":"Test message","details":null}}"#;
        let error = parse_error_response(json).unwrap();
        assert_eq!(error.code, "TEST_ERROR");
        assert_eq!(error.message, "Test message");
    }

    #[tokio::test]
    async fn test_is_port_available() {
        // Port 0 should always be available (OS assigns)
        assert!(is_port_available(0).await);
    }
}
