//! Generic daemon handle for lifecycle management
//!
//! TEAM-315: Extracted from queen-lifecycle, made generic

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Generic daemon handle for lifecycle management
///
/// Tracks whether we started the daemon and provides cleanup.
/// IMPORTANT: Only shuts down daemon if we started it!
///
/// # Example
///
/// ```rust
/// use daemon_contract::DaemonHandle;
///
/// // Daemon already running
/// let handle = DaemonHandle::already_running("queen-rbee", "http://localhost:7833");
/// assert!(!handle.should_cleanup());
///
/// // We started the daemon
/// let handle = DaemonHandle::started_by_us("rbee-hive", "http://localhost:7835", Some(12345));
/// assert!(handle.should_cleanup());
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DaemonHandle {
    /// Daemon name (e.g., "queen-rbee", "rbee-hive", "vllm-worker")
    daemon_name: String,
    
    /// True if we started the daemon (must cleanup)
    /// False if daemon was already running (don't touch it)
    started_by_us: bool,

    /// Base URL of the daemon
    base_url: String,

    /// Process ID if we started it
    #[serde(skip_serializing_if = "Option::is_none")]
    pid: Option<u32>,
}

impl DaemonHandle {
    /// Create handle for daemon that was already running
    ///
    /// # Arguments
    /// * `daemon_name` - Name of the daemon
    /// * `base_url` - Base URL of the daemon
    pub fn already_running(daemon_name: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            daemon_name: daemon_name.into(),
            started_by_us: false,
            base_url: base_url.into(),
            pid: None,
        }
    }

    /// Create handle for daemon that we just started
    ///
    /// # Arguments
    /// * `daemon_name` - Name of the daemon
    /// * `base_url` - Base URL of the daemon
    /// * `pid` - Process ID (if available)
    pub fn started_by_us(
        daemon_name: impl Into<String>,
        base_url: impl Into<String>,
        pid: Option<u32>,
    ) -> Self {
        Self {
            daemon_name: daemon_name.into(),
            started_by_us: true,
            base_url: base_url.into(),
            pid,
        }
    }

    /// Check if we started the daemon (and should clean it up)
    pub const fn should_cleanup(&self) -> bool {
        self.started_by_us
    }

    /// Get the daemon's base URL
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
    
    /// Get the daemon name
    pub fn daemon_name(&self) -> &str {
        &self.daemon_name
    }
    
    /// Get the process ID (if we started it)
    pub const fn pid(&self) -> Option<u32> {
        self.pid
    }
    
    /// Update the handle with discovered URL
    ///
    /// Service discovery - update URL after fetching from /v1/info
    pub fn with_discovered_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Keep the daemon alive (no shutdown after task)
    ///
    /// Daemon stays running for future tasks.
    pub fn shutdown(self) -> Result<()> {
        // Note: Actual shutdown logic is in lifecycle crates
        // This just indicates we're done with the handle
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_already_running() {
        let handle = DaemonHandle::already_running("test-daemon", "http://localhost:8080");
        assert!(!handle.should_cleanup());
        assert_eq!(handle.daemon_name(), "test-daemon");
        assert_eq!(handle.base_url(), "http://localhost:8080");
        assert_eq!(handle.pid(), None);
    }

    #[test]
    fn test_started_by_us() {
        let handle = DaemonHandle::started_by_us("test-daemon", "http://localhost:8080", Some(12345));
        assert!(handle.should_cleanup());
        assert_eq!(handle.daemon_name(), "test-daemon");
        assert_eq!(handle.base_url(), "http://localhost:8080");
        assert_eq!(handle.pid(), Some(12345));
    }

    #[test]
    fn test_with_discovered_url() {
        let handle = DaemonHandle::already_running("test-daemon", "http://localhost:8080")
            .with_discovered_url("http://192.168.1.100:8080");
        assert_eq!(handle.base_url(), "http://192.168.1.100:8080");
    }

    #[test]
    fn test_serialization() {
        let handle = DaemonHandle::started_by_us("test-daemon", "http://localhost:8080", Some(12345));
        let json = serde_json::to_string(&handle).unwrap();
        let deserialized: DaemonHandle = serde_json::from_str(&json).unwrap();
        assert_eq!(handle.daemon_name(), deserialized.daemon_name());
        assert_eq!(handle.base_url(), deserialized.base_url());
        assert_eq!(handle.pid(), deserialized.pid());
    }
}
