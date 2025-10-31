//! Hive registry implementation
//!
//! TEAM-284: Thread-safe registry for tracking hive state
//! TEAM-285: Migrated to use generic HeartbeatRegistry

use heartbeat_registry::HeartbeatRegistry;
use hive_contract::{HiveHeartbeat, HiveInfo};
use std::collections::HashMap;
use std::sync::RwLock;

// TEAM-362: Worker telemetry storage
use rbee_hive_monitor::ProcessStats;

/// Telemetry registry
///
/// TEAM-374: Renamed from HiveRegistry to TelemetryRegistry for clarity.
/// Stores both hive heartbeats AND worker telemetry (sent by hives).
///
/// Thread-safe registry using RwLock for concurrent access.
/// Hives send telemetry via POST /v1/hive-heartbeat or SSE streams.
///
/// # Example
///
/// ```
/// use queen_rbee_telemetry_registry::TelemetryRegistry;
/// use hive_contract::{HiveInfo, HiveHeartbeat};
/// use shared_contract::{OperationalStatus, HealthStatus};
///
/// let registry = TelemetryRegistry::new();
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
pub struct TelemetryRegistry {
    inner: HeartbeatRegistry<HiveHeartbeat>,
    
    // TEAM-362: Worker telemetry storage (hive_id -> workers)
    workers: RwLock<HashMap<String, Vec<ProcessStats>>>,
}

impl TelemetryRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            inner: HeartbeatRegistry::new(),
            workers: RwLock::new(HashMap::new()),  // TEAM-362: Worker storage
        }
    }

    /// Update hive from heartbeat
    ///
    /// Upserts hive info - creates if new, updates if exists.
    pub fn update_hive(&self, heartbeat: HiveHeartbeat) {
        self.inner.update(heartbeat);
    }

    /// Get hive by ID
    pub fn get_hive(&self, hive_id: &str) -> Option<HiveInfo> {
        self.inner.get(hive_id)
    }

    /// Remove hive from registry
    ///
    /// Returns true if hive was removed, false if not found.
    pub fn remove_hive(&self, hive_id: &str) -> bool {
        self.inner.remove(hive_id)
    }

    /// List all hives (including stale ones)
    pub fn list_all_hives(&self) -> Vec<HiveInfo> {
        self.inner.list_all()
    }

    /// List hives with recent heartbeats
    ///
    /// Only returns hives that sent heartbeat within timeout window.
    pub fn list_online_hives(&self) -> Vec<HiveInfo> {
        self.inner.list_online()
    }

    /// List available hives (online + ready status)
    ///
    /// Returns hives that are:
    /// 1. Online (recent heartbeat)
    /// 2. Ready status (not busy/starting/stopped)
    pub fn list_available_hives(&self) -> Vec<HiveInfo> {
        self.inner.list_available()
    }

    /// Get count of online hives
    pub fn count_online(&self) -> usize {
        self.inner.count_online()
    }

    /// Get count of available hives
    pub fn count_available(&self) -> usize {
        self.inner.count_available()
    }

    // TEAM-362: Worker telemetry methods
    
    /// Store workers for a hive
    pub fn update_workers(&self, hive_id: &str, workers: Vec<ProcessStats>) {
        let mut map = self.workers.write().unwrap();
        map.insert(hive_id.to_string(), workers);
    }
    
    /// Get workers for a hive
    pub fn get_workers(&self, hive_id: &str) -> Option<Vec<ProcessStats>> {
        let map = self.workers.read().unwrap();
        map.get(hive_id).cloned()
    }
    
    /// Get all workers across all hives
    pub fn get_all_workers(&self) -> Vec<ProcessStats> {
        let map = self.workers.read().unwrap();
        map.values().flatten().cloned().collect()
    }
    
    /// Find idle workers (gpu_util_pct == 0.0)
    pub fn find_idle_workers(&self) -> Vec<ProcessStats> {
        self.get_all_workers()
            .into_iter()
            .filter(|w| w.gpu_util_pct == 0.0)
            .collect()
    }
    
    /// Find workers with specific model loaded
    pub fn find_workers_with_model(&self, model: &str) -> Vec<ProcessStats> {
        self.get_all_workers()
            .into_iter()
            .filter(|w| w.model.as_deref() == Some(model))
            .collect()
    }
    
    /// Find workers with available VRAM capacity
    /// TEAM-364: Now uses worker's actual total_vram_mb instead of hardcoded limit (Critical Issue #5)
    pub fn find_workers_with_capacity(&self, required_vram_mb: u64) -> Vec<ProcessStats> {
        self.get_all_workers()
            .into_iter()
            .filter(|w| {
                // TEAM-364: Use worker's actual total VRAM (queried from nvidia-smi)
                // Falls back to 24GB if not available
                let total_vram = if w.total_vram_mb > 0 { w.total_vram_mb } else { 24576 };
                w.vram_mb + required_vram_mb < total_vram
            })
            .collect()
    }
    
    /// Find best worker for model
    ///
    /// TEAM-374: Added for scheduler compatibility (was in WorkerRegistry)
    /// Finds the best idle worker for a given model:
    /// 1. Must have the model loaded
    /// 2. Must be idle (gpu_util_pct == 0.0)
    /// 3. Prefers worker with lowest load
    pub fn find_best_worker_for_model(&self, model: &str) -> Option<ProcessStats> {
        self.find_idle_workers()
            .into_iter()
            .find(|w| w.model.as_deref() == Some(model))
    }
    
    /// List online workers (compatibility method)
    ///
    /// TEAM-374: Added for compatibility with old WorkerRegistry API
    /// Returns ProcessStats for all workers (not WorkerInfo)
    pub fn list_online_workers(&self) -> Vec<ProcessStats> {
        self.get_all_workers()
    }

    /// Cleanup stale hives
    ///
    /// Removes hives that haven't sent heartbeat within timeout window.
    /// Returns number of hives removed.
    pub fn cleanup_stale(&self) -> usize {
        self.inner.cleanup_stale()
    }
}

impl Default for TelemetryRegistry {
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
        let registry = TelemetryRegistry::new();
        assert_eq!(registry.list_all_hives().len(), 0);
    }

    #[test]
    fn registry_update_hive() {
        let registry = TelemetryRegistry::new();
        let hive = create_hive("hive-1", OperationalStatus::Ready);
        let heartbeat = HiveHeartbeat::new(hive);

        registry.update_hive(heartbeat);

        assert_eq!(registry.list_all_hives().len(), 1);
        assert!(registry.get_hive("hive-1").is_some());
    }

    #[test]
    fn registry_get_hive() {
        let registry = TelemetryRegistry::new();
        let hive = create_hive("hive-1", OperationalStatus::Ready);
        registry.update_hive(HiveHeartbeat::new(hive));

        let retrieved = registry.get_hive("hive-1").unwrap();
        assert_eq!(retrieved.id, "hive-1");
        assert_eq!(retrieved.port, 9200);
    }

    #[test]
    fn registry_remove_hive() {
        let registry = TelemetryRegistry::new();
        let hive = create_hive("hive-1", OperationalStatus::Ready);
        registry.update_hive(HiveHeartbeat::new(hive));

        assert!(registry.remove_hive("hive-1"));
        assert!(!registry.remove_hive("hive-1")); // Already removed
        assert_eq!(registry.list_all_hives().len(), 0);
    }

    #[test]
    fn registry_list_online_hives() {
        let registry = TelemetryRegistry::new();

        // Add recent hive
        let hive1 = create_hive("hive-1", OperationalStatus::Ready);
        registry.update_hive(HiveHeartbeat::new(hive1));

        // Add old hive
        let hive2 = create_hive("hive-2", OperationalStatus::Ready);
        let old_timestamp = HeartbeatTimestamp::from_datetime(Utc::now() - Duration::seconds(120));
        let old_heartbeat = HiveHeartbeat { hive: hive2, timestamp: old_timestamp, workers: Vec::new() };
        registry.update_hive(old_heartbeat);

        let online = registry.list_online_hives();
        assert_eq!(online.len(), 1);
        assert_eq!(online[0].id, "hive-1");
    }

    #[test]
    fn registry_list_available_hives() {
        let registry = TelemetryRegistry::new();

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
        let registry = TelemetryRegistry::new();

        let hive1 = create_hive("hive-1", OperationalStatus::Ready);
        let hive2 = create_hive("hive-2", OperationalStatus::Ready);

        registry.update_hive(HiveHeartbeat::new(hive1));
        registry.update_hive(HiveHeartbeat::new(hive2));

        assert_eq!(registry.count_online(), 2);
    }

    #[test]
    fn registry_cleanup_stale() {
        let registry = TelemetryRegistry::new();

        // Add recent hive
        let hive1 = create_hive("hive-1", OperationalStatus::Ready);
        registry.update_hive(HiveHeartbeat::new(hive1));

        // Add old hive
        let hive2 = create_hive("hive-2", OperationalStatus::Ready);
        let old_timestamp = HeartbeatTimestamp::from_datetime(Utc::now() - Duration::seconds(120));
        let old_heartbeat = HiveHeartbeat { hive: hive2, timestamp: old_timestamp, workers: Vec::new() };
        registry.update_hive(old_heartbeat);

        let removed = registry.cleanup_stale();
        assert_eq!(removed, 1);
        assert_eq!(registry.list_all_hives().len(), 1);
    }
}
