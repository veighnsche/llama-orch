//! queen-rbee-worker-registry
//!
//! In-memory registry for tracking real-time runtime state of all workers.
//!
//! TEAM-221: Investigated 2025-10-22 - Comprehensive behavior inventory complete
//! TEAM-261: Simplified - workers send heartbeats directly to queen
//! TEAM-262: Renamed from hive-registry to worker-registry
//!
//! This is DIFFERENT from `hive-catalog` (SQLite - persistent storage):
//! - **Catalog** = Persistent config (host, port, SSH, device capabilities)
//! - **Registry** = Runtime state (workers, VRAM usage, last heartbeat)
//!
//! ## Usage
//!
//! ```rust
//! use queen_rbee_worker_registry::WorkerRegistry;
//! use rbee_heartbeat::WorkerHeartbeatPayload;
//!
//! let registry = WorkerRegistry::new();
//!
//! // Update from heartbeat
//! let payload = HiveHeartbeatPayload {
//!     hive_id: "localhost".to_string(),
//!     timestamp: "2025-10-21T10:00:00Z".to_string(),
//!     workers: vec![],
//! };
//! registry.update_hive_state("localhost", payload);
//!
//! // Check if hive is online
//! let is_online = registry.is_hive_online("localhost", 30_000);
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

mod types;

pub use types::{HiveRuntimeState, ResourceInfo, WorkerInfo};

use rbee_heartbeat::HiveHeartbeatPayload;
use std::collections::HashMap;
use std::sync::RwLock;

/// In-memory registry for tracking worker runtime state
///
/// Thread-safe registry using RwLock for concurrent access.
/// Optimized for read-heavy workload (scheduling queries).
///
/// TEAM-262: Renamed from HiveRegistry - tracks workers directly after TEAM-261
/// TEAM-262: Struct name changed, internal logic unchanged
pub struct WorkerRegistry {
    hives: RwLock<HashMap<String, HiveRuntimeState>>,
}

impl WorkerRegistry {
    /// Create a new empty hive registry
    pub fn new() -> Self {
        Self { hives: RwLock::new(HashMap::new()) }
    }

    /// Update hive runtime state from heartbeat
    ///
    /// Processes incoming heartbeat and updates:
    /// - Worker list
    /// - Last heartbeat timestamp
    /// - VRAM/RAM usage (calculated from workers)
    /// - Worker count
    pub fn update_hive_state(&self, hive_id: &str, payload: HiveHeartbeatPayload) {
        // Parse timestamp
        let timestamp_ms = chrono::DateTime::parse_from_rfc3339(&payload.timestamp)
            .map(|dt| dt.timestamp_millis())
            .unwrap_or_else(|_| chrono::Utc::now().timestamp_millis());

        // Convert WorkerState to WorkerInfo
        let workers: Vec<WorkerInfo> = payload
            .workers
            .into_iter()
            .map(|w| WorkerInfo {
                worker_id: w.worker_id,
                state: w.state,
                last_heartbeat: w.last_heartbeat,
                health_status: w.health_status,
                url: w.url,
                model_id: w.model_id,
                backend: w.backend,
                device_id: w.device_id,
                vram_bytes: w.vram_bytes,
                ram_bytes: w.ram_bytes,
                cpu_percent: w.cpu_percent,
                gpu_percent: w.gpu_percent,
            })
            .collect();

        // Create runtime state
        let state = HiveRuntimeState::from_heartbeat(hive_id.to_string(), workers, timestamp_ms);

        // Update registry
        let mut hives = self.hives.write().unwrap();
        hives.insert(hive_id.to_string(), state);
    }

    /// Get current runtime state for a hive
    ///
    /// Returns None if hive not found in registry.
    pub fn get_hive_state(&self, hive_id: &str) -> Option<HiveRuntimeState> {
        let hives = self.hives.read().unwrap();
        hives.get(hive_id).cloned()
    }

    /// List all hives that sent heartbeat recently
    ///
    /// # Parameters
    /// - `max_age_ms`: Maximum age of last heartbeat (e.g., 30000 = 30 seconds)
    ///
    /// # Returns
    /// List of hive IDs that are considered "online"
    pub fn list_active_hives(&self, max_age_ms: i64) -> Vec<String> {
        let hives = self.hives.read().unwrap();

        hives
            .iter()
            .filter(|(_, state)| state.is_recent(max_age_ms))
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Get available resources for a hive
    ///
    /// Returns None if hive not found.
    pub fn get_available_resources(&self, hive_id: &str) -> Option<ResourceInfo> {
        let hives = self.hives.read().unwrap();
        hives.get(hive_id).map(|state| state.resource_info())
    }

    /// Remove hive from registry
    ///
    /// Returns true if hive was removed, false if not found.
    pub fn remove_hive(&self, hive_id: &str) -> bool {
        let mut hives = self.hives.write().unwrap();
        hives.remove(hive_id).is_some()
    }

    /// List all hive IDs in registry
    ///
    /// Returns all hives regardless of heartbeat age.
    pub fn list_all_hives(&self) -> Vec<String> {
        let hives = self.hives.read().unwrap();
        hives.keys().cloned().collect()
    }

    /// Get worker count for a hive
    ///
    /// Returns None if hive not found.
    pub fn get_worker_count(&self, hive_id: &str) -> Option<usize> {
        let hives = self.hives.read().unwrap();
        hives.get(hive_id).map(|state| state.worker_count)
    }

    /// Check if hive is online (recent heartbeat)
    ///
    /// # Parameters
    /// - `hive_id`: Hive to check
    /// - `max_age_ms`: Maximum age of last heartbeat (e.g., 30000 = 30 seconds)
    pub fn is_hive_online(&self, hive_id: &str, max_age_ms: i64) -> bool {
        let hives = self.hives.read().unwrap();
        hives.get(hive_id).map(|state| state.is_recent(max_age_ms)).unwrap_or(false)
    }

    /// Get total number of hives in registry
    pub fn hive_count(&self) -> usize {
        let hives = self.hives.read().unwrap();
        hives.len()
    }

    // ========================================================================
    // WORKER REGISTRY FUNCTIONS
    // ========================================================================
    // The hive-registry also serves as the worker registry.
    // All worker info comes from heartbeats and is stored here.

    /// Get worker by ID (searches across all hives)
    ///
    /// Returns (hive_id, worker_info) if found.
    pub fn get_worker(&self, worker_id: &str) -> Option<(String, WorkerInfo)> {
        let hives = self.hives.read().unwrap();

        for (hive_id, state) in hives.iter() {
            if let Some(worker) = state.workers.iter().find(|w| w.worker_id == worker_id) {
                return Some((hive_id.clone(), worker.clone()));
            }
        }
        None
    }

    /// Get worker URL for direct inference
    ///
    /// This is the primary way to get a worker's URL for routing inference requests.
    pub fn get_worker_url(&self, worker_id: &str) -> Option<String> {
        self.get_worker(worker_id).map(|(_, worker)| worker.url)
    }

    /// List all workers across all hives
    ///
    /// Returns list of (hive_id, worker_info) tuples.
    pub fn list_all_workers(&self) -> Vec<(String, WorkerInfo)> {
        let hives = self.hives.read().unwrap();

        hives
            .iter()
            .flat_map(|(hive_id, state)| {
                state.workers.iter().map(move |w| (hive_id.clone(), w.clone()))
            })
            .collect()
    }

    /// Find idle workers (state == "Idle")
    ///
    /// Returns list of (hive_id, worker_info) for workers that are idle.
    pub fn find_idle_workers(&self) -> Vec<(String, WorkerInfo)> {
        self.list_all_workers().into_iter().filter(|(_, worker)| worker.state == "Idle").collect()
    }

    /// Find workers by model
    ///
    /// Returns workers that have the specified model loaded.
    pub fn find_workers_by_model(&self, model_id: &str) -> Vec<(String, WorkerInfo)> {
        self.list_all_workers()
            .into_iter()
            .filter(|(_, worker)| worker.model_id.as_ref().map(|m| m == model_id).unwrap_or(false))
            .collect()
    }

    /// Find workers by backend
    ///
    /// Returns workers using the specified backend (e.g., "cuda", "cpu").
    pub fn find_workers_by_backend(&self, backend: &str) -> Vec<(String, WorkerInfo)> {
        self.list_all_workers()
            .into_iter()
            .filter(|(_, worker)| worker.backend.as_ref().map(|b| b == backend).unwrap_or(false))
            .collect()
    }

    /// Find best worker for model
    ///
    /// Finds the best idle worker for a given model:
    /// 1. Prefers workers with model already loaded
    /// 2. Among those, prefers workers with lowest GPU usage
    /// 3. Falls back to any idle worker with lowest GPU usage
    pub fn find_best_worker_for_model(&self, model_id: &str) -> Option<(String, WorkerInfo)> {
        let idle_workers = self.find_idle_workers();

        // First, try to find idle workers with model already loaded
        let with_model: Vec<_> = idle_workers
            .iter()
            .filter(|(_, w)| w.model_id.as_ref().map(|m| m == model_id).unwrap_or(false))
            .collect();

        if !with_model.is_empty() {
            // Find one with lowest GPU usage
            return with_model
                .into_iter()
                .min_by(|(_, a), (_, b)| {
                    let a_gpu = a.gpu_percent.unwrap_or(100.0);
                    let b_gpu = b.gpu_percent.unwrap_or(100.0);
                    a_gpu.partial_cmp(&b_gpu).unwrap()
                })
                .map(|(hive, worker)| (hive.clone(), worker.clone()));
        }

        // Fall back to any idle worker with lowest GPU usage
        idle_workers.into_iter().min_by(|(_, a), (_, b)| {
            let a_gpu = a.gpu_percent.unwrap_or(100.0);
            let b_gpu = b.gpu_percent.unwrap_or(100.0);
            a_gpu.partial_cmp(&b_gpu).unwrap()
        })
    }

    /// Get total worker count across all hives
    pub fn total_worker_count(&self) -> usize {
        let hives = self.hives.read().unwrap();
        hives.values().map(|state| state.worker_count).sum()
    }

    /// Get workers on a specific hive
    pub fn get_workers_on_hive(&self, hive_id: &str) -> Vec<WorkerInfo> {
        let hives = self.hives.read().unwrap();
        hives.get(hive_id).map(|state| state.workers.clone()).unwrap_or_default()
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
    use rbee_heartbeat::types::WorkerState;

    fn create_test_payload(hive_id: &str, worker_count: usize) -> HiveHeartbeatPayload {
        let workers: Vec<WorkerState> = (0..worker_count)
            .map(|i| WorkerState {
                worker_id: format!("worker-{}", i),
                state: "Idle".to_string(),
                last_heartbeat: "2025-10-21T10:00:00Z".to_string(),
                health_status: "healthy".to_string(),
                url: format!("http://localhost:{}", 9300 + i),
                model_id: if i % 2 == 0 { Some("llama-3-8b".to_string()) } else { None },
                backend: Some("cuda".to_string()),
                device_id: Some(i as u32),
                vram_bytes: Some(8_000_000_000),
                ram_bytes: Some(2_000_000_000),
                cpu_percent: Some(10.0 + i as f32),
                gpu_percent: Some(5.0 * i as f32),
            })
            .collect();

        HiveHeartbeatPayload {
            hive_id: hive_id.to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            workers,
        }
    }

    #[test]
    fn test_new_registry_is_empty() {
        let registry = HiveRegistry::new();
        assert_eq!(registry.hive_count(), 0);
        assert_eq!(registry.list_all_hives().len(), 0);
    }

    #[test]
    fn test_update_hive_state() {
        let registry = HiveRegistry::new();
        let payload = create_test_payload("localhost", 2);

        registry.update_hive_state("localhost", payload);

        assert_eq!(registry.hive_count(), 1);
        assert!(registry.get_hive_state("localhost").is_some());
    }

    #[test]
    fn test_get_hive_state() {
        let registry = HiveRegistry::new();
        let payload = create_test_payload("localhost", 3);

        registry.update_hive_state("localhost", payload);

        let state = registry.get_hive_state("localhost").unwrap();
        assert_eq!(state.hive_id, "localhost");
        assert_eq!(state.worker_count, 3);
        assert_eq!(state.workers.len(), 3);
    }

    #[test]
    fn test_get_hive_state_not_found() {
        let registry = HiveRegistry::new();
        assert!(registry.get_hive_state("nonexistent").is_none());
    }

    #[test]
    fn test_list_active_hives() {
        let registry = HiveRegistry::new();

        // Add hive with recent heartbeat
        let payload = create_test_payload("hive1", 1);
        registry.update_hive_state("hive1", payload);

        // Check active hives (30 second window)
        let active = registry.list_active_hives(30_000);
        assert_eq!(active.len(), 1);
        assert!(active.contains(&"hive1".to_string()));
    }

    #[test]
    fn test_list_active_hives_filters_old() {
        let registry = HiveRegistry::new();

        // Add hive with old heartbeat
        let mut payload = create_test_payload("old-hive", 1);
        payload.timestamp = "2020-01-01T00:00:00Z".to_string(); // Very old

        registry.update_hive_state("old-hive", payload);

        // Should not appear in active list
        let active = registry.list_active_hives(30_000);
        assert_eq!(active.len(), 0);
    }

    #[test]
    fn test_get_available_resources() {
        let registry = HiveRegistry::new();
        let payload = create_test_payload("localhost", 4);

        registry.update_hive_state("localhost", payload);

        let resources = registry.get_available_resources("localhost").unwrap();
        assert_eq!(resources.worker_count, 4);
        assert_eq!(resources.vram_used_gb, 16.0); // 4 workers * 4GB
        assert_eq!(resources.ram_used_gb, 8.0); // 4 workers * 2GB
    }

    #[test]
    fn test_remove_hive() {
        let registry = HiveRegistry::new();
        let payload = create_test_payload("localhost", 1);

        registry.update_hive_state("localhost", payload);
        assert_eq!(registry.hive_count(), 1);

        let removed = registry.remove_hive("localhost");
        assert!(removed);
        assert_eq!(registry.hive_count(), 0);
    }

    #[test]
    fn test_remove_hive_not_found() {
        let registry = HiveRegistry::new();
        let removed = registry.remove_hive("nonexistent");
        assert!(!removed);
    }

    #[test]
    fn test_list_all_hives() {
        let registry = HiveRegistry::new();

        registry.update_hive_state("hive1", create_test_payload("hive1", 1));
        registry.update_hive_state("hive2", create_test_payload("hive2", 2));
        registry.update_hive_state("hive3", create_test_payload("hive3", 3));

        let all_hives = registry.list_all_hives();
        assert_eq!(all_hives.len(), 3);
        assert!(all_hives.contains(&"hive1".to_string()));
        assert!(all_hives.contains(&"hive2".to_string()));
        assert!(all_hives.contains(&"hive3".to_string()));
    }

    #[test]
    fn test_get_worker_count() {
        let registry = HiveRegistry::new();
        let payload = create_test_payload("localhost", 5);

        registry.update_hive_state("localhost", payload);

        let count = registry.get_worker_count("localhost").unwrap();
        assert_eq!(count, 5);
    }

    #[test]
    fn test_get_worker_count_not_found() {
        let registry = HiveRegistry::new();
        assert!(registry.get_worker_count("nonexistent").is_none());
    }

    #[test]
    fn test_is_hive_online_true() {
        let registry = HiveRegistry::new();
        let payload = create_test_payload("localhost", 1);

        registry.update_hive_state("localhost", payload);

        assert!(registry.is_hive_online("localhost", 30_000));
    }

    #[test]
    fn test_is_hive_online_false_old_heartbeat() {
        let registry = HiveRegistry::new();
        let mut payload = create_test_payload("localhost", 1);
        payload.timestamp = "2020-01-01T00:00:00Z".to_string();

        registry.update_hive_state("localhost", payload);

        assert!(!registry.is_hive_online("localhost", 30_000));
    }

    #[test]
    fn test_is_hive_online_false_not_found() {
        let registry = HiveRegistry::new();
        assert!(!registry.is_hive_online("nonexistent", 30_000));
    }

    #[test]
    fn test_update_existing_hive() {
        let registry = HiveRegistry::new();

        // First heartbeat
        registry.update_hive_state("localhost", create_test_payload("localhost", 2));
        assert_eq!(registry.get_worker_count("localhost").unwrap(), 2);

        // Second heartbeat with different worker count
        registry.update_hive_state("localhost", create_test_payload("localhost", 5));
        assert_eq!(registry.get_worker_count("localhost").unwrap(), 5);
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let registry = Arc::new(HiveRegistry::new());
        let mut handles = vec![];

        // Spawn multiple threads updating different hives
        for i in 0..10 {
            let registry_clone = Arc::clone(&registry);
            let handle = thread::spawn(move || {
                let hive_id = format!("hive-{}", i);
                let payload = create_test_payload(&hive_id, i);
                registry_clone.update_hive_state(&hive_id, payload);
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify all hives were added
        assert_eq!(registry.hive_count(), 10);
    }

    // ========================================================================
    // WORKER REGISTRY TESTS
    // ========================================================================

    #[test]
    fn test_get_worker() {
        let registry = HiveRegistry::new();
        registry.update_hive_state("hive1", create_test_payload("hive1", 3));

        let (hive_id, worker) = registry.get_worker("worker-1").unwrap();
        assert_eq!(hive_id, "hive1");
        assert_eq!(worker.worker_id, "worker-1");
        assert_eq!(worker.url, "http://localhost:9301");
    }

    #[test]
    fn test_get_worker_not_found() {
        let registry = HiveRegistry::new();
        assert!(registry.get_worker("nonexistent").is_none());
    }

    #[test]
    fn test_get_worker_url() {
        let registry = HiveRegistry::new();
        registry.update_hive_state("hive1", create_test_payload("hive1", 2));

        let url = registry.get_worker_url("worker-0").unwrap();
        assert_eq!(url, "http://localhost:9300");
    }

    #[test]
    fn test_list_all_workers() {
        let registry = HiveRegistry::new();
        registry.update_hive_state("hive1", create_test_payload("hive1", 2));
        registry.update_hive_state("hive2", create_test_payload("hive2", 3));

        let all_workers = registry.list_all_workers();
        assert_eq!(all_workers.len(), 5); // 2 + 3
    }

    #[test]
    fn test_find_idle_workers() {
        let registry = HiveRegistry::new();
        registry.update_hive_state("hive1", create_test_payload("hive1", 3));

        let idle = registry.find_idle_workers();
        assert_eq!(idle.len(), 3); // All are idle in test data
    }

    #[test]
    fn test_find_workers_by_model() {
        let registry = HiveRegistry::new();
        registry.update_hive_state("hive1", create_test_payload("hive1", 4));

        let with_model = registry.find_workers_by_model("llama-3-8b");
        assert_eq!(with_model.len(), 2); // Even indices have model
    }

    #[test]
    fn test_find_workers_by_backend() {
        let registry = HiveRegistry::new();
        registry.update_hive_state("hive1", create_test_payload("hive1", 3));

        let cuda_workers = registry.find_workers_by_backend("cuda");
        assert_eq!(cuda_workers.len(), 3); // All use cuda in test data
    }

    #[test]
    fn test_find_best_worker_for_model() {
        let registry = HiveRegistry::new();
        registry.update_hive_state("hive1", create_test_payload("hive1", 4));

        let best = registry.find_best_worker_for_model("llama-3-8b").unwrap();
        // Should prefer worker with model loaded and lowest GPU usage
        assert_eq!(best.1.model_id, Some("llama-3-8b".to_string()));
        // worker-0 has gpu_percent = 0.0, worker-2 has gpu_percent = 10.0
        assert_eq!(best.1.worker_id, "worker-0");
    }

    #[test]
    fn test_total_worker_count() {
        let registry = HiveRegistry::new();
        registry.update_hive_state("hive1", create_test_payload("hive1", 2));
        registry.update_hive_state("hive2", create_test_payload("hive2", 3));

        assert_eq!(registry.total_worker_count(), 5);
    }

    #[test]
    fn test_get_workers_on_hive() {
        let registry = HiveRegistry::new();
        registry.update_hive_state("hive1", create_test_payload("hive1", 3));
        registry.update_hive_state("hive2", create_test_payload("hive2", 2));

        let hive1_workers = registry.get_workers_on_hive("hive1");
        assert_eq!(hive1_workers.len(), 3);

        let hive2_workers = registry.get_workers_on_hive("hive2");
        assert_eq!(hive2_workers.len(), 2);
    }

    #[test]
    fn test_get_workers_on_nonexistent_hive() {
        let registry = HiveRegistry::new();
        let workers = registry.get_workers_on_hive("nonexistent");
        assert_eq!(workers.len(), 0);
    }
}
