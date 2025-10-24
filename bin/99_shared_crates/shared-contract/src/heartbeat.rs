//! Heartbeat protocol definitions
//!
//! TEAM-284: Common heartbeat types and traits

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Heartbeat timestamp wrapper
///
/// Provides consistent timestamp handling across all heartbeats.
/// Uses ISO 8601 format for serialization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeartbeatTimestamp(pub DateTime<Utc>);

impl HeartbeatTimestamp {
    /// Create a new timestamp with current time
    pub fn now() -> Self {
        Self(Utc::now())
    }
    
    /// Create from DateTime
    pub fn from_datetime(dt: DateTime<Utc>) -> Self {
        Self(dt)
    }
    
    /// Get the inner DateTime
    pub fn inner(&self) -> &DateTime<Utc> {
        &self.0
    }
    
    /// Check if timestamp is recent (within timeout window)
    pub fn is_recent(&self, timeout_secs: u64) -> bool {
        let now = Utc::now();
        let elapsed = now.signed_duration_since(self.0);
        elapsed.num_seconds() < timeout_secs as i64
    }
    
    /// Get age in seconds
    pub fn age_secs(&self) -> i64 {
        let now = Utc::now();
        now.signed_duration_since(self.0).num_seconds()
    }
}

/// Trait for heartbeat payloads
///
/// All heartbeat types (worker, hive) must implement this trait.
pub trait HeartbeatPayload: Serialize + for<'de> Deserialize<'de> + Clone {
    /// Get the component ID (worker_id or hive_id)
    fn component_id(&self) -> &str;
    
    /// Get the heartbeat timestamp
    fn timestamp(&self) -> &HeartbeatTimestamp;
    
    /// Check if heartbeat is recent
    fn is_recent(&self, timeout_secs: u64) -> bool {
        self.timestamp().is_recent(timeout_secs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Duration;

    #[test]
    fn timestamp_now() {
        let ts = HeartbeatTimestamp::now();
        assert!(ts.is_recent(60));
    }

    #[test]
    fn timestamp_is_recent() {
        let now = Utc::now();
        let recent = HeartbeatTimestamp::from_datetime(now - Duration::seconds(30));
        let old = HeartbeatTimestamp::from_datetime(now - Duration::seconds(120));

        assert!(recent.is_recent(60));
        assert!(!old.is_recent(60));
    }

    #[test]
    fn timestamp_age() {
        let now = Utc::now();
        let ts = HeartbeatTimestamp::from_datetime(now - Duration::seconds(45));
        
        let age = ts.age_secs();
        assert!(age >= 44 && age <= 46); // Allow 1s tolerance
    }

    #[test]
    fn timestamp_serialization() {
        let ts = HeartbeatTimestamp::now();
        let json = serde_json::to_string(&ts).unwrap();
        
        // Should serialize as ISO 8601 string
        assert!(json.contains("T"));
        assert!(json.contains("Z"));
        
        // Should deserialize back
        let deserialized: HeartbeatTimestamp = serde_json::from_str(&json).unwrap();
        assert_eq!(ts, deserialized);
    }
}
