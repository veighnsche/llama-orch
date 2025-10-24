//! Hive registry implementation
//!
//! TEAM-284: Thread-safe registry for tracking hive state

use hive_contract::{HiveHeartbeat, HiveInfo};
use std::collections::HashMap;
use std::sync::RwLock;

/// Hive registry
///
/// Thread-safe registry using RwLock for concurrent access.
/// Hives send heartbeats directly to queen via POST /v1/hive-heartbeat.
///
/// # Example
///
/// ```
/// use queen_rbee_hive_registry::HiveRegistry;
/// use hive_contract::{HiveInfo, HiveHeartbeat};
/// use shared_contract::{OperationalStatus, HealthStatus};
///
/// let registry = HiveRegistry::new();
///
/// // Hive sends heartbeat
/// let hive = HiveInfo {
///     id: "localhost".to_string(),
///     hostname: "127.0.0.1".to_string(),
///     port: 9200,
///     operational_status: OperationalStatus::Ready,
///     health_status: HealthStatus::Healthy,
///     version: "0.1.0".to_string(),
/// };
///
/// registry.update_hive(HiveHeartbeat::new(hive));
///
/// // Query hives
/// let available = registry.list_available_hives();
/// ```
pub struct HiveRegistry {
    hives: RwLock<HashMap<String, HiveHeartbeat>>,
}

impl HiveRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            hives: RwLock::new(HashMap::new()),
        }
    }

    /// Update hive from heartbeat
    ///
    /// Upserts hive info - creates if new, updates if exists.
    pub fn update_hive(&self, heartbeat: HiveHeartbeat) {
        let mut hives = self.hives.write().unwrap();
        hives.insert(heartbeat.hive.id.clone(), heartbeat);
    }

    /// Get hive by ID
    pub fn get_hive(&self, hive_id: &str) -> Option<HiveInfo> {
        let hives = self.hives.read().unwrap();
        hives.get(hive_id).map(|hb| hb.hive.clone())
    }

    /// Remove hive from registry
    ///
    /// Returns true if hive was removed, false if not found.
    pub fn remove_hive(&self, hive_id: &str) -> bool {
        let mut hives = self.hives.write().unwrap();
        hives.remove(hive_id).is_some()
    }

    /// List all hives (including stale ones)
    pub fn list_all_hives(&self) -> Vec<HiveInfo> {
        let hives = self.hives.read().unwrap();
        hives.values().map(|hb| hb.hive.clone()).collect()
    }

    /// List hives with recent heartbeats
    ///
    /// Only returns hives that sent heartbeat within timeout window.
    pub fn list_online_hives(&self) -> Vec<HiveInfo> {
        let hives = self.hives.read().unwrap();
        hives
            .values()
            .filter(|hb| hb.is_recent())
            .map(|hb| hb.hive.clone())
            .collect()
    }

    /// List available hives (online + ready status)
    ///
    /// Returns hives that are:
    /// 1. Online (recent heartbeat)
    /// 2. Ready status (not busy/starting/stopped)
    pub fn list_available_hives(&self) -> Vec<HiveInfo> {
        let hives = self.hives.read().unwrap();
        hives
            .values()
            .filter(|hb| hb.is_recent() && hb.hive.is_available())
            .map(|hb| hb.hive.clone())
            .collect()
    }

    /// Get count of online hives
    pub fn count_online(&self) -> usize {
        let hives = self.hives.read().unwrap();
        hives.values().filter(|hb| hb.is_recent()).count()
    }

    /// Get count of available hives
    pub fn count_available(&self) -> usize {
        let hives = self.hives.read().unwrap();
        hives
            .values()
            .filter(|hb| hb.is_recent() && hb.hive.is_available())
            .count()
    }

    /// Cleanup stale hives
    ///
    /// Removes hives that haven't sent heartbeat within timeout window.
    /// Returns number of hives removed.
    pub fn cleanup_stale(&self) -> usize {
        let mut hives = self.hives.write().unwrap();
        let before_count = hives.len();
        hives.retain(|_, hb| hb.is_recent());
        before_count - hives.len()
    }
}

impl Default for HiveRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Duration, Utc};
    use shared_contract::{HealthStatus, HeartbeatTimestamp, OperationalStatus};

    fn create_hive(id: &str, status: OperationalStatus) -> HiveInfo {
        HiveInfo {
            id: id.to_string(),
            hostname: "localhost".to_string(),
            port: 9200,
            operational_status: status,
            health_status: HealthStatus::Healthy,
            version: "0.1.0".to_string(),
        }
    }

    #[test]
    fn registry_new() {
        let registry = HiveRegistry::new();
        assert_eq!(registry.list_all_hives().len(), 0);
    }

    #[test]
    fn registry_update_hive() {
        let registry = HiveRegistry::new();
        let hive = create_hive("hive-1", OperationalStatus::Ready);
        let heartbeat = HiveHeartbeat::new(hive);

        registry.update_hive(heartbeat);

        assert_eq!(registry.list_all_hives().len(), 1);
        assert!(registry.get_hive("hive-1").is_some());
    }

    #[test]
    fn registry_get_hive() {
        let registry = HiveRegistry::new();
        let hive = create_hive("hive-1", OperationalStatus::Ready);
        registry.update_hive(HiveHeartbeat::new(hive));

        let retrieved = registry.get_hive("hive-1").unwrap();
        assert_eq!(retrieved.id, "hive-1");
        assert_eq!(retrieved.port, 9200);
    }

    #[test]
    fn registry_remove_hive() {
        let registry = HiveRegistry::new();
        let hive = create_hive("hive-1", OperationalStatus::Ready);
        registry.update_hive(HiveHeartbeat::new(hive));

        assert!(registry.remove_hive("hive-1"));
        assert!(!registry.remove_hive("hive-1")); // Already removed
        assert_eq!(registry.list_all_hives().len(), 0);
    }

    #[test]
    fn registry_list_online_hives() {
        let registry = HiveRegistry::new();

        // Add recent hive
        let hive1 = create_hive("hive-1", OperationalStatus::Ready);
        registry.update_hive(HiveHeartbeat::new(hive1));

        // Add old hive
        let hive2 = create_hive("hive-2", OperationalStatus::Ready);
        let old_timestamp = HeartbeatTimestamp::from_datetime(Utc::now() - Duration::seconds(120));
        let old_heartbeat = HiveHeartbeat {
            hive: hive2,
            timestamp: old_timestamp,
        };
        registry.update_hive(old_heartbeat);

        let online = registry.list_online_hives();
        assert_eq!(online.len(), 1);
        assert_eq!(online[0].id, "hive-1");
    }

    #[test]
    fn registry_list_available_hives() {
        let registry = HiveRegistry::new();

        // Ready hive
        let hive1 = create_hive("hive-1", OperationalStatus::Ready);
        registry.update_hive(HiveHeartbeat::new(hive1));

        // Busy hive (still available)
        let hive2 = create_hive("hive-2", OperationalStatus::Busy);
        registry.update_hive(HiveHeartbeat::new(hive2));

        // Stopped hive (not available)
        let hive3 = create_hive("hive-3", OperationalStatus::Stopped);
        registry.update_hive(HiveHeartbeat::new(hive3));

        let available = registry.list_available_hives();
        assert_eq!(available.len(), 2);
    }

    #[test]
    fn registry_count_online() {
        let registry = HiveRegistry::new();

        let hive1 = create_hive("hive-1", OperationalStatus::Ready);
        let hive2 = create_hive("hive-2", OperationalStatus::Ready);

        registry.update_hive(HiveHeartbeat::new(hive1));
        registry.update_hive(HiveHeartbeat::new(hive2));

        assert_eq!(registry.count_online(), 2);
    }

    #[test]
    fn registry_cleanup_stale() {
        let registry = HiveRegistry::new();

        // Add recent hive
        let hive1 = create_hive("hive-1", OperationalStatus::Ready);
        registry.update_hive(HiveHeartbeat::new(hive1));

        // Add old hive
        let hive2 = create_hive("hive-2", OperationalStatus::Ready);
        let old_timestamp = HeartbeatTimestamp::from_datetime(Utc::now() - Duration::seconds(120));
        let old_heartbeat = HiveHeartbeat {
            hive: hive2,
            timestamp: old_timestamp,
        };
        registry.update_hive(old_heartbeat);

        let removed = registry.cleanup_stale();
        assert_eq!(removed, 1);
        assert_eq!(registry.list_all_hives().len(), 1);
    }
}
