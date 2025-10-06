//! Integration test framework
//!
//! Provides test harness for spawning and managing worker processes
//! during integration tests.
//!
//! # Spec References
//! - M0-W-1820: Integration test framework

use std::net::TcpListener;
use std::process::{Child, Command};
use std::time::Duration;
use tokio::time::timeout;
use worker_http::validation::ExecuteRequest;

/// Error type for test framework
#[derive(Debug, thiserror::Error)]
pub enum TestError {
    #[error("Failed to spawn worker process: {0}")]
    SpawnFailed(String),

    #[error("Worker failed to become ready within timeout")]
    ReadyTimeout,

    #[error("HTTP request failed: {0}")]
    HttpFailed(String),

    #[error("Worker process died unexpectedly")]
    ProcessDied,

    #[error("Invalid response: {0}")]
    InvalidResponse(String),
}

/// Test harness for worker-orcd integration tests
///
/// Manages worker process lifecycle:
/// - Spawns worker with test configuration
/// - Waits for readiness
/// - Provides HTTP client helpers
/// - Automatic cleanup on drop
///
/// # Example
///
/// ```no_run
/// use worker_orcd::tests::integration::WorkerTestHarness;
///
/// #[tokio::test]
/// async fn test_inference() {
///     let harness = WorkerTestHarness::start_mock().await.unwrap();
///     
///     let response = harness.health().await.unwrap();
///     assert_eq!(response.status, "healthy");
/// }
/// ```
pub struct WorkerTestHarness {
    process: Option<Child>,
    port: u16,
    worker_id: String,
    base_url: String,
}

impl WorkerTestHarness {
    /// Start worker with real model
    ///
    /// Spawns worker process with specified model and GPU device.
    /// Waits up to 30 seconds for worker to become ready.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to model file (.gguf)
    /// * `gpu_device` - GPU device ID (0-based)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Worker binary not found
    /// - Worker fails to start
    /// - Worker doesn't become ready within timeout
    pub async fn start(model_path: &str, gpu_device: i32) -> Result<Self, TestError> {
        let port = find_free_port();
        let worker_id = format!("test-worker-{}", port);

        // Try release build first, fall back to debug
        // Use CARGO_BIN_EXE or find in target directory
        let binary_path = std::env::var("CARGO_BIN_EXE_worker-orcd").unwrap_or_else(|_| {
            // Check workspace root target directories
            let workspace_root =
                std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
            let release_path = format!("{}/../../target/release/worker-orcd", workspace_root);
            let debug_path = format!("{}/../../target/debug/worker-orcd", workspace_root);

            if std::path::Path::new(&release_path).exists() {
                release_path
            } else {
                debug_path
            }
        });

        let process = Command::new(&binary_path)
            .args(&[
                "--worker-id",
                &worker_id,
                "--model",
                model_path,
                "--gpu-device",
                &gpu_device.to_string(),
                "--port",
                &port.to_string(),
                "--callback-url",
                "http://localhost:9999/callback", // Dummy callback for testing
            ])
            .spawn()
            .map_err(|e| TestError::SpawnFailed(format!("{} (binary: {})", e, binary_path)))?;

        let base_url = format!("http://localhost:{}", port);

        let mut harness = Self { process: Some(process), port, worker_id, base_url };

        harness.wait_for_ready(Duration::from_secs(30)).await?;

        Ok(harness)
    }

    /// Start worker with mock/stub mode (no real model)
    ///
    /// For fast tests that don't need actual inference.
    /// Worker runs in stub mode with mock responses.
    pub async fn start_mock() -> Result<Self, TestError> {
        let port = find_free_port();
        let worker_id = format!("test-worker-mock-{}", port);

        let process = Command::new("target/debug/worker-orcd")
            .args(&[
                "--worker-id",
                &worker_id,
                "--port",
                &port.to_string(),
                "--stub-mode", // Run without CUDA
            ])
            .spawn()
            .map_err(|e| TestError::SpawnFailed(e.to_string()))?;

        let base_url = format!("http://localhost:{}", port);

        let mut harness = Self { process: Some(process), port, worker_id, base_url };

        harness.wait_for_ready(Duration::from_secs(60)).await?; // Increased for FP16 model loading

        Ok(harness)
    }

    /// Wait for worker to become ready
    ///
    /// Polls health endpoint until it responds or timeout expires.
    async fn wait_for_ready(&mut self, timeout_duration: Duration) -> Result<(), TestError> {
        let start = std::time::Instant::now();

        loop {
            // Check if process died
            if let Some(ref mut process) = self.process {
                if let Ok(Some(_)) = process.try_wait() {
                    return Err(TestError::ProcessDied);
                }
            }

            // Try health endpoint
            match self.health().await {
                Ok(_) => return Ok(()),
                Err(_) => {
                    // Not ready yet
                    if start.elapsed() > timeout_duration {
                        return Err(TestError::ReadyTimeout);
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }
    }

    /// Send execute request
    ///
    /// Returns SSE event stream for processing.
    pub async fn execute(&mut self, req: ExecuteRequest) -> Result<reqwest::Response, TestError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120)) // Long timeout for inference
            .build()
            .map_err(|e| TestError::HttpFailed(e.to_string()))?;

        // Check if process is still alive
        if let Some(ref mut process) = self.process {
            if let Ok(Some(status)) = process.try_wait() {
                eprintln!("❌ Worker process died with status: {:?}", status);
                return Err(TestError::ProcessDied);
            } else {
                eprintln!("✅ Worker process is still running");
            }
        }

        eprintln!("🔍 Sending POST to {}/execute", self.base_url);

        let response =
            client.post(format!("{}/execute", self.base_url)).json(&req).send().await.map_err(
                |e| {
                    eprintln!("❌ Request failed: {}", e);
                    TestError::HttpFailed(e.to_string())
                },
            )?;

        eprintln!("✅ Got response with status: {}", response.status());
        Ok(response)
    }

    /// Check health endpoint
    pub async fn health(&self) -> Result<serde_json::Value, TestError> {
        let client = reqwest::Client::new();

        let response =
            timeout(Duration::from_secs(5), client.get(format!("{}/health", self.base_url)).send())
                .await
                .map_err(|_| TestError::ReadyTimeout)?
                .map_err(|e| TestError::HttpFailed(e.to_string()))?;

        if !response.status().is_success() {
            return Err(TestError::InvalidResponse(format!(
                "Health check failed with status {}",
                response.status()
            )));
        }

        response.json().await.map_err(|e| TestError::InvalidResponse(e.to_string()))
    }

    /// Cancel inference job
    pub async fn cancel(&self, job_id: &str) -> Result<(), TestError> {
        let client = reqwest::Client::new();

        let response = client
            .post(format!("{}/cancel/{}", self.base_url, job_id))
            .send()
            .await
            .map_err(|e| TestError::HttpFailed(e.to_string()))?;

        if !response.status().is_success() {
            return Err(TestError::InvalidResponse(format!(
                "Cancel failed with status {}",
                response.status()
            )));
        }

        Ok(())
    }

    /// Get base URL
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Get worker ID
    pub fn worker_id(&self) -> &str {
        &self.worker_id
    }

    /// Get port
    pub fn port(&self) -> u16 {
        self.port
    }
}

impl Drop for WorkerTestHarness {
    fn drop(&mut self) {
        if let Some(mut process) = self.process.take() {
            let _ = process.kill();
            let _ = process.wait();
        }
    }
}

/// Find available port for testing
fn find_free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("Failed to bind to port 0");
    let port = listener.local_addr().expect("Failed to get local addr").port();
    drop(listener);
    port
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_free_port() {
        let port1 = find_free_port();
        let port2 = find_free_port();

        assert!(port1 > 0);
        assert!(port2 > 0);
        assert_ne!(port1, port2);
    }

    #[tokio::test]
    async fn test_harness_cleanup() {
        // Create harness without starting (just test Drop)
        let port = find_free_port();
        let harness = WorkerTestHarness {
            process: None,
            port,
            worker_id: "test".to_string(),
            base_url: format!("http://localhost:{}", port),
        };

        drop(harness);
        // Should not panic
    }
}

// ---
// Built by Foundation-Alpha 🏗️
