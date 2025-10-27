//! SSH target types
//!
//! TEAM-315: Extracted from ssh-config to eliminate duplication

use serde::{Deserialize, Serialize};

use crate::status::SshTargetStatus;

/// SSH target from ~/.ssh/config
///
/// Represents a host entry from SSH configuration with connection details.
///
/// # Example
///
/// ```rust
/// use ssh_contract::{SshTarget, SshTargetStatus};
///
/// let target = SshTarget {
///     host: "workstation".to_string(),
///     host_subtitle: None,
///     hostname: "192.168.1.100".to_string(),
///     user: "vince".to_string(),
///     port: 22,
///     status: SshTargetStatus::Unknown,
/// };
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SshTarget {
    /// Host alias from SSH config (first word)
    pub host: String,
    
    /// Host subtitle (second word, optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub host_subtitle: Option<String>,
    
    /// Hostname (IP or domain)
    pub hostname: String,
    
    /// SSH username
    pub user: String,
    
    /// SSH port
    pub port: u16,
    
    /// Connection status
    pub status: SshTargetStatus,
}

impl SshTarget {
    /// Create a new SSH target
    pub fn new(
        host: impl Into<String>,
        hostname: impl Into<String>,
        user: impl Into<String>,
        port: u16,
    ) -> Self {
        Self {
            host: host.into(),
            host_subtitle: None,
            hostname: hostname.into(),
            user: user.into(),
            port,
            status: SshTargetStatus::Unknown,
        }
    }

    /// Create with subtitle
    pub fn with_subtitle(mut self, subtitle: impl Into<String>) -> Self {
        self.host_subtitle = Some(subtitle.into());
        self
    }

    /// Update connection status
    pub fn with_status(mut self, status: SshTargetStatus) -> Self {
        self.status = status;
        self
    }

    /// Get full SSH connection string (user@hostname:port)
    pub fn connection_string(&self) -> String {
        format!("{}@{}:{}", self.user, self.hostname, self.port)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ssh_target_creation() {
        let target = SshTarget::new("workstation", "192.168.1.100", "vince", 22);
        assert_eq!(target.host, "workstation");
        assert_eq!(target.hostname, "192.168.1.100");
        assert_eq!(target.user, "vince");
        assert_eq!(target.port, 22);
        assert_eq!(target.status, SshTargetStatus::Unknown);
    }

    #[test]
    fn test_connection_string() {
        let target = SshTarget::new("workstation", "192.168.1.100", "vince", 2222);
        assert_eq!(target.connection_string(), "vince@192.168.1.100:2222");
    }

    #[test]
    fn test_serialization() {
        let target = SshTarget::new("workstation", "192.168.1.100", "vince", 22)
            .with_status(SshTargetStatus::Online);
        
        let json = serde_json::to_string(&target).unwrap();
        let deserialized: SshTarget = serde_json::from_str(&json).unwrap();
        
        assert_eq!(target, deserialized);
    }
}
