//! SSH target connection status
//!
//! TEAM-315: Extracted from ssh-config
//! TEAM-316: Added optional Tauri/Specta support

use serde::{Deserialize, Serialize};

// TEAM-316: Optional Tauri/Specta support for TypeScript generation
#[cfg(feature = "tauri")]
use specta::Type;

/// SSH target connection status
///
/// Indicates whether an SSH host is reachable.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[cfg_attr(feature = "tauri", derive(Type))]
#[serde(rename_all = "lowercase")]
pub enum SshTargetStatus {
    /// Host is reachable and responding
    Online,
    
    /// Host is unreachable or not responding
    Offline,
    
    /// Status has not been checked yet
    Unknown,
}

impl Default for SshTargetStatus {
    fn default() -> Self {
        Self::Unknown
    }
}

impl SshTargetStatus {
    /// Check if host is online
    pub const fn is_online(&self) -> bool {
        matches!(self, Self::Online)
    }

    /// Check if host is offline
    pub const fn is_offline(&self) -> bool {
        matches!(self, Self::Offline)
    }

    /// Check if status is unknown
    pub const fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown)
    }

    /// Get status as string
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Online => "online",
            Self::Offline => "offline",
            Self::Unknown => "unknown",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_checks() {
        assert!(SshTargetStatus::Online.is_online());
        assert!(SshTargetStatus::Offline.is_offline());
        assert!(SshTargetStatus::Unknown.is_unknown());
    }

    #[test]
    fn test_status_as_str() {
        assert_eq!(SshTargetStatus::Online.as_str(), "online");
        assert_eq!(SshTargetStatus::Offline.as_str(), "offline");
        assert_eq!(SshTargetStatus::Unknown.as_str(), "unknown");
    }

    #[test]
    fn test_serialization() {
        let status = SshTargetStatus::Online;
        let json = serde_json::to_string(&status).unwrap();
        assert_eq!(json, "\"online\"");
        
        let deserialized: SshTargetStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(status, deserialized);
    }
}
