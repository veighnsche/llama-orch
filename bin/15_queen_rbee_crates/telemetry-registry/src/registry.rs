//! Hive registry implementation
//!
//! TEAM-284: Thread-safe registry for tracking hive state
//! TEAM-377: RULE ZERO - Removed heartbeat-based tracking
//!           Connection state is source of truth, not timestamps

use hive_contract::HiveInfo;
use std::collections::HashMap;
use std::sync::RwLock;

// TEAM-362: Worker telemetry storage
use rbee_hive_monitor::ProcessStats;

/// Telemetry registry
///
/// TEAM-377: BREAKING CHANGE - Connection-based tracking
/// 
/// Stores hives that have active SSE connections.
/// No timestamps, no timeouts - connection state IS the source of truth.
///
/// Thread-safe registry using RwLock for concurrent access.
///
/// # Example
///
/// ```
/// use queen_rbee_telemetry_registry::TelemetryRegistry;
/// use hive_contract::HiveInfo;
/// use shared_contract::{OperationalStatus, HealthStatus};
///
/// let registry = TelemetryRegistry::new();
///
/// // Register hive when SSE connection opens
/// let hive = HiveInfo {
///     id: "localhost".to_string(),
///     hostname: "127.0.0.1".to_string(),
///     port: 9200,
///     operational_status: OperationalStatus::Ready,
///     health_status: HealthStatus::Healthy,
///     version: "0.1.0".to_string(),
/// };
///
/// registry.register_hive(hive);
///
/// // Remove hive when SSE connection closes
/// registry.remove_hive("localhost");
/// ```
pub struct TelemetryRegistry {
    // TEAM-377: Just store HiveInfo, not HiveHeartbeat
    // If it's in the map, it's online (has active connection)
    hives: RwLock<HashMap<String, HiveInfo>>,
    
    // TEAM-362: Worker telemetry storage (hive_id -> workers)
    workers: RwLock<HashMap<String, Vec<ProcessStats>>>,
}

impl TelemetryRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            hives: RwLock::new(HashMap::new()),
            workers: RwLock::new(HashMap::new()),
        }
    }

    /// TEAM-377: Register hive when SSE connection opens
    ///
    /// Call this when Queen establishes SSE connection to hive.
    /// Replaces update_hive() - no timestamps needed.
    pub fn register_hive(&self, hive_info: HiveInfo) {
        let mut hives = self.hives.write().unwrap();
        hives.insert(hive_info.id.clone(), hive_info);
    }

    /// Get hive by ID
    pub fn get_hive(&self, hive_id: &str) -> Option<HiveInfo> {
        let hives = self.hives.read().unwrap();
        hives.get(hive_id).cloned()
    }

    /// TEAM-377: Remove hive when SSE connection closes
    ///
    /// Call this immediately when connection fails/closes.
    /// No timeout needed - connection state is the source of truth.
    ///
    /// Returns true if hive was removed, false if not found.
    pub fn remove_hive(&self, hive_id: &str) -> bool {
        let mut hives = self.hives.write().unwrap();
        hives.remove(hive_id).is_some()
    }

    /// TEAM-377: List all online hives
    ///
    /// Returns all hives with active SSE connections.
    /// If it's in the map, it's online.
    pub fn list_online_hives(&self) -> Vec<HiveInfo> {
        let hives = self.hives.read().unwrap();
        hives.values().cloned().collect()
    }

    /// TEAM-377: Get count of online hives
    ///
    /// Returns number of hives with active SSE connections.
    /// No filtering needed - if it's in the map, it's online.
    pub fn count_online(&self) -> usize {
        let hives = self.hives.read().unwrap();
        hives.len()
    }

    /// Get count of available hives (same as online for SSE connections)
    pub fn count_available(&self) -> usize {
        self.count_online()
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
    
    // TEAM-377: DELETED cleanup_stale() - not needed with connection-based tracking
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
