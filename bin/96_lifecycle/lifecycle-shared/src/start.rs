//! Start configuration shared between lifecycle-local and lifecycle-ssh
//!
//! TEAM-367: Extracted from lifecycle-local/src/start.rs and lifecycle-ssh/src/start.rs

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for HTTP-based daemons
///
/// TEAM-330: Moved from types/start.rs (RULE ZERO - inline it)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpDaemonConfig {
    /// Daemon name for narration (e.g., "queen-rbee", "rbee-hive")
    pub daemon_name: String,

    /// Health check URL (e.g., "http://localhost:7833/health")
    pub health_url: String,

    /// Optional job ID for narration routing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub job_id: Option<String>,

    /// Path to daemon binary (optional - auto-resolved from daemon_name if not provided)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub binary_path: Option<PathBuf>,

    /// Command-line arguments for daemon
    #[serde(default)]
    pub args: Vec<String>,

    /// Maximum health check attempts (default: 10)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_health_attempts: Option<usize>,

    /// Initial health check delay in ms (default: 200)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub health_initial_delay_ms: Option<u64>,

    /// Process ID (required for signal-based shutdown)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,

    /// Graceful shutdown timeout in seconds (default: 5)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graceful_timeout_secs: Option<u64>,

    /// TEAM-359: Monitoring group (e.g., "llm", "queen", "hive")
    /// Used for cgroup organization on Linux
    #[serde(skip_serializing_if = "Option::is_none")]
    pub monitor_group: Option<String>,

    /// TEAM-359: Monitoring instance (e.g., port number)
    /// Used for cgroup organization on Linux
    #[serde(skip_serializing_if = "Option::is_none")]
    pub monitor_instance: Option<String>,
}

impl HttpDaemonConfig {
    /// Create new HttpDaemonConfig
    pub fn new(daemon_name: impl Into<String>, health_url: impl Into<String>) -> Self {
        Self {
            daemon_name: daemon_name.into(),
            health_url: health_url.into(),
            job_id: None,
            binary_path: None,
            args: Vec::new(),
            max_health_attempts: None,
            health_initial_delay_ms: None,
            pid: None,
            graceful_timeout_secs: None,
            monitor_group: None,
            monitor_instance: None,
        }
    }

    /// Set job ID for narration routing
    pub fn with_job_id(mut self, job_id: impl Into<String>) -> Self {
        self.job_id = Some(job_id.into());
        self
    }

    /// Set binary path
    pub fn with_binary_path(mut self, path: PathBuf) -> Self {
        self.binary_path = Some(path);
        self
    }

    /// Set command-line arguments
    pub fn with_args(mut self, args: Vec<String>) -> Self {
        self.args = args;
        self
    }

    /// Set maximum health check attempts
    pub fn with_max_health_attempts(mut self, attempts: usize) -> Self {
        self.max_health_attempts = Some(attempts);
        self
    }

    /// Set initial health check delay
    pub fn with_health_initial_delay_ms(mut self, delay_ms: u64) -> Self {
        self.health_initial_delay_ms = Some(delay_ms);
        self
    }

    /// Set process ID
    pub fn with_pid(mut self, pid: u32) -> Self {
        self.pid = Some(pid);
        self
    }

    /// Set graceful shutdown timeout
    pub fn with_graceful_timeout_secs(mut self, timeout_secs: u64) -> Self {
        self.graceful_timeout_secs = Some(timeout_secs);
        self
    }

    /// TEAM-359: Set monitoring group
    pub fn with_monitor_group(mut self, group: impl Into<String>) -> Self {
        self.monitor_group = Some(group.into());
        self
    }

    /// TEAM-359: Set monitoring instance
    pub fn with_monitor_instance(mut self, instance: impl Into<String>) -> Self {
        self.monitor_instance = Some(instance.into());
        self
    }
}

/// Generate shell command to find daemon binary
///
/// TEAM-367: Shared logic extracted from lifecycle-local and lifecycle-ssh
///
/// Searches in order:
/// 1. target/debug/{daemon} (development builds)
/// 2. target/release/{daemon} (release builds)
/// 3. ~/.local/bin/{daemon} (installed binaries)
/// 4. which {daemon} (system PATH)
///
/// Returns "NOT_FOUND" if binary not found
pub fn find_binary_command(daemon_name: &str) -> String {
    // TEAM-377: RULE ZERO - Use constant for install directory
    use crate::BINARY_INSTALL_DIR;
    
    format!(
        "(test -x target/debug/{} && echo target/debug/{}) || \
         (test -x target/release/{} && echo target/release/{}) || \
         (test -x ~/{}/{} && echo ~/{}/{}) || \
         which {} 2>/dev/null || \
         echo 'NOT_FOUND'",
        daemon_name, daemon_name, 
        daemon_name, daemon_name, 
        BINARY_INSTALL_DIR, daemon_name, BINARY_INSTALL_DIR, daemon_name,
        daemon_name
    )
}

/// Generate shell command to start daemon in background
///
/// TEAM-367: Shared logic extracted from lifecycle-local and lifecycle-ssh
///
/// Uses nohup to detach from terminal and captures PID
///
/// # Example
/// ```rust,ignore
/// let cmd = build_start_command("/usr/local/bin/rbee-hive", &["--port", "7835"]);
/// // Returns: "nohup /usr/local/bin/rbee-hive --port 7835 > /dev/null 2>&1 & echo $!"
/// ```
pub fn build_start_command(binary_path: &str, args: &[String]) -> String {
    if args.is_empty() {
        format!("nohup {} > /dev/null 2>&1 & echo $!", binary_path)
    } else {
        let args_str = args.join(" ");
        format!("nohup {} {} > /dev/null 2>&1 & echo $!", binary_path, args_str)
    }
}
