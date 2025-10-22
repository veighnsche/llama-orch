// TEAM-251: Test harness for integration tests
// Purpose: Spawn actual binaries, manage lifecycle, capture output, validate state

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};
use tempfile::TempDir;
use tokio::time::sleep;
use uuid::Uuid;

/// Result of running a command
#[derive(Debug, Clone)]
pub struct CommandResult {
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

/// Process state
#[derive(Debug, Clone, PartialEq)]
pub enum ProcessState {
    Stopped,
    Running,
}

/// System state (all processes)
#[derive(Debug, Clone, PartialEq)]
pub struct SystemState {
    pub queen: ProcessState,
    pub hive: ProcessState,
}

/// Test harness for integration tests
pub struct TestHarness {
    /// Temporary directory for test isolation
    temp_dir: TempDir,
    /// Running processes
    processes: HashMap<String, Child>,
    /// Queen port (dynamically allocated)
    queen_port: u16,
    /// Hive port (dynamically allocated)
    hive_port: u16,
    /// Test ID for debugging
    test_id: String,
}

impl TestHarness {
    /// Create new isolated test environment
    pub async fn new() -> Result<Self> {
        let temp_dir = TempDir::new().context("Failed to create temp directory")?;
        let test_id = Uuid::new_v4().to_string();

        // TEAM-255: Use default ports since binaries don't respect custom port env vars
        // This means tests cannot run in parallel, but they will work
        let queen_port = 8500;  // Default queen port from queen-rbee/src/main.rs
        let hive_port = 9000;   // Default hive port from rbee-hive/src/main.rs

        // Set up isolated environment
        env::set_var("RBEE_CONFIG_DIR", temp_dir.path().join("config"));
        env::set_var("RBEE_DATA_DIR", temp_dir.path().join("data"));
        env::set_var("RBEE_TEST_ID", &test_id);

        // Create config directory
        std::fs::create_dir_all(temp_dir.path().join("config"))?;
        std::fs::create_dir_all(temp_dir.path().join("data"))?;

        Ok(Self { temp_dir, processes: HashMap::new(), queen_port, hive_port, test_id })
    }

    /// Find a free port
    fn find_free_port() -> Result<u16> {
        use std::net::TcpListener;

        let listener = TcpListener::bind("127.0.0.1:0")?;
        let port = listener.local_addr()?.port();
        drop(listener);
        Ok(port)
    }

    /// Find binary path
    fn find_binary(&self, name: &str) -> Result<PathBuf> {
        // TEAM-255: Find workspace root by looking for Cargo.toml
        let mut current = env::current_dir()?;
        let workspace_root = loop {
            if current.join("Cargo.toml").exists() && current.join("xtask").exists() {
                break current;
            }
            if let Some(parent) = current.parent() {
                current = parent.to_path_buf();
            } else {
                bail!("Could not find workspace root");
            }
        };

        // Try debug first, then release
        let debug_path = workspace_root.join("target/debug").join(name);
        if debug_path.exists() {
            return Ok(debug_path);
        }

        let release_path = workspace_root.join("target/release").join(name);
        if release_path.exists() {
            return Ok(release_path);
        }

        bail!("Binary '{}' not found. Run: cargo build --bin {}", name, name)
    }

    /// Run a command and capture output
    pub async fn run_command(&mut self, cmd: &[&str]) -> Result<CommandResult> {
        let binary = self.find_binary("rbee-keeper")?;

        println!("🔧 Running: {} {}", binary.display(), cmd.join(" "));

        // TEAM-255: Pass environment variables to child process
        let output = Command::new(binary)
            .args(cmd)
            .env("RBEE_CONFIG_DIR", self.temp_dir.path().join("config"))
            .env("RBEE_DATA_DIR", self.temp_dir.path().join("data"))
            .env("RBEE_TEST_ID", &self.test_id)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .context("Failed to execute command")?;

        let result = CommandResult {
            exit_code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        };

        println!("✅ Exit code: {:?}", result.exit_code);
        if !result.stdout.is_empty() {
            println!("📤 Stdout:\n{}", result.stdout);
        }
        if !result.stderr.is_empty() {
            println!("📤 Stderr:\n{}", result.stderr);
        }

        Ok(result)
    }

    /// Check if process is running via health endpoint
    pub fn is_running(&self, name: &str) -> bool {
        let port = match name {
            "queen" => self.queen_port,
            "hive" => self.hive_port,
            _ => return false,
        };

        let url = format!("http://127.0.0.1:{}/health", port);

        match ureq::get(&url).timeout(Duration::from_secs(1)).call() {
            Ok(response) => response.status() == 200,
            Err(_) => false,
        }
    }

    /// Wait for process to be ready
    pub async fn wait_for_ready(&self, name: &str, timeout: Duration) -> Result<()> {
        let start = Instant::now();

        println!("⏳ Waiting for {} to be ready (timeout: {:?})...", name, timeout);

        loop {
            if self.is_running(name) {
                println!("✅ {} is ready", name);
                return Ok(());
            }

            if start.elapsed() > timeout {
                bail!("Timeout waiting for {} to be ready", name);
            }

            sleep(Duration::from_millis(100)).await;
        }
    }

    /// Get current system state
    pub async fn get_state(&self) -> SystemState {
        SystemState {
            queen: if self.is_running("queen") {
                ProcessState::Running
            } else {
                ProcessState::Stopped
            },
            hive: if self.is_running("hive") {
                ProcessState::Running
            } else {
                ProcessState::Stopped
            },
        }
    }

    /// Set system to specific state
    pub async fn set_state(&mut self, state: SystemState) -> Result<()> {
        let current = self.get_state().await;

        // Start queen if needed
        if state.queen == ProcessState::Running && current.queen == ProcessState::Stopped {
            self.run_command(&["queen", "start"]).await?;
            self.wait_for_ready("queen", Duration::from_secs(10)).await?;
        }

        // Stop queen if needed
        if state.queen == ProcessState::Stopped && current.queen == ProcessState::Running {
            self.run_command(&["queen", "stop"]).await?;
            sleep(Duration::from_secs(1)).await;
        }

        // Start hive if needed
        if state.hive == ProcessState::Running && current.hive == ProcessState::Stopped {
            self.run_command(&["hive", "start"]).await?;
            self.wait_for_ready("hive", Duration::from_secs(10)).await?;
        }

        // Stop hive if needed
        if state.hive == ProcessState::Stopped && current.hive == ProcessState::Running {
            self.run_command(&["hive", "stop"]).await?;
            sleep(Duration::from_secs(1)).await;
        }

        Ok(())
    }

    /// Kill a specific process
    pub async fn kill_process(&mut self, name: &str) -> Result<()> {
        println!("💀 Killing process: {}", name);

        // Use pkill to kill by name
        let _ = Command::new("pkill").args(["-9", "-f", name]).output();

        sleep(Duration::from_millis(500)).await;
        Ok(())
    }

    /// Clean up all processes (async version with delay)
    pub async fn cleanup(&mut self) -> Result<()> {
        println!("🧹 Cleaning up test environment (ID: {})...", self.test_id);

        // TEAM-252: DO NOT use run_command() here - it requires finding binaries!
        // If we're in a panic context (binary not found), run_command() will hang.
        // Instead, directly kill processes using pkill.

        // Force kill any processes by name (no binary lookup needed)
        let _ = Command::new("pkill").args(["-9", "-f", "queen-rbee"]).output();
        let _ = Command::new("pkill").args(["-9", "-f", "rbee-hive"]).output();
        let _ = Command::new("pkill").args(["-9", "-f", "rbee-keeper"]).output();

        // Kill all tracked processes
        for (name, mut child) in self.processes.drain() {
            println!("🔪 Killing tracked process: {}", name);
            let _ = child.kill();
        }

        // Brief wait for processes to die
        sleep(Duration::from_millis(200)).await;

        println!("✅ Cleanup complete");
        Ok(())
    }

    /// Synchronous cleanup for Drop handler
    /// TEAM-252: Drop cannot use async/await or block_on() - causes deadlock in tokio tests!
    fn cleanup_sync(&mut self) {
        // Force kill any processes by name (no binary lookup needed)
        let _ = Command::new("pkill").args(["-9", "-f", "queen-rbee"]).output();
        let _ = Command::new("pkill").args(["-9", "-f", "rbee-hive"]).output();
        let _ = Command::new("pkill").args(["-9", "-f", "rbee-keeper"]).output();

        // Kill all tracked processes
        for (_name, mut child) in self.processes.drain() {
            let _ = child.kill();
        }

        // Synchronous sleep
        std::thread::sleep(Duration::from_millis(200));
    }

    /// Get temp directory path
    pub fn temp_dir(&self) -> &Path {
        self.temp_dir.path()
    }
}

impl Drop for TestHarness {
    fn drop(&mut self) {
        // TEAM-252: Use synchronous cleanup to avoid deadlock
        // block_on() creates a new runtime which deadlocks inside tokio tests
        self.cleanup_sync();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_harness_creation() {
        let harness = TestHarness::new().await.unwrap();
        assert!(harness.temp_dir().exists());
        assert!(harness.queen_port > 0);
        assert!(harness.hive_port > 0);
        assert_ne!(harness.queen_port, harness.hive_port);
    }

    #[tokio::test]
    async fn test_find_binary() {
        let harness = TestHarness::new().await.unwrap();
        let binary = harness.find_binary("rbee-keeper");
        assert!(binary.is_ok());
    }
}
