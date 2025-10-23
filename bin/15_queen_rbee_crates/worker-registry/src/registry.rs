// TEAM-270: Worker registry (workers send heartbeats to queen)

use std::collections::HashMap;
use std::sync::RwLock;
use worker_contract::{WorkerHeartbeat, WorkerInfo};

/// Worker registry
///
/// Thread-safe registry using RwLock for concurrent access.
/// Workers send heartbeats directly to queen via POST /v1/worker-heartbeat.
///
/// # Example
///
/// ```
/// use queen_rbee_worker_registry::WorkerRegistry;
/// use worker_contract::{WorkerInfo, WorkerStatus, WorkerHeartbeat};
///
/// let registry = WorkerRegistry::new();
///
/// // Worker sends heartbeat
/// let worker = WorkerInfo {
///     id: "worker-123".to_string(),
///     model_id: "meta-llama/Llama-2-7b".to_string(),
///     device: "GPU-0".to_string(),
///     port: 9301,
///     status: WorkerStatus::Ready,
///     implementation: "llm-worker-rbee".to_string(),
///     version: "0.1.0".to_string(),
/// };
///
/// registry.update_worker(WorkerHeartbeat::new(worker));
///
/// // Query workers
/// let available = registry.list_available_workers();
/// ```
pub struct WorkerRegistry {
    workers: RwLock<HashMap<String, WorkerHeartbeat>>,
}

impl WorkerRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self { workers: RwLock::new(HashMap::new()) }
    }

    /// Update worker from heartbeat
    ///
    /// Upserts worker info - creates if new, updates if exists.
    pub fn update_worker(&self, heartbeat: WorkerHeartbeat) {
        let mut workers = self.workers.write().unwrap();
        workers.insert(heartbeat.worker.id.clone(), heartbeat);
    }

    /// Get worker by ID
    pub fn get_worker(&self, worker_id: &str) -> Option<WorkerInfo> {
        let workers = self.workers.read().unwrap();
        workers.get(worker_id).map(|hb| hb.worker.clone())
    }

    /// Remove worker from registry
    ///
    /// Returns true if worker was removed, false if not found.
    pub fn remove_worker(&self, worker_id: &str) -> bool {
        let mut workers = self.workers.write().unwrap();
        workers.remove(worker_id).is_some()
    }

    /// List all workers (including stale ones)
    pub fn list_all_workers(&self) -> Vec<WorkerInfo> {
        let workers = self.workers.read().unwrap();
        workers.values().map(|hb| hb.worker.clone()).collect()
    }

    /// List workers with recent heartbeats
    ///
    /// Only returns workers that sent heartbeat within timeout window.
    pub fn list_online_workers(&self) -> Vec<WorkerInfo> {
        let workers = self.workers.read().unwrap();
        workers.values().filter(|hb| hb.is_recent()).map(|hb| hb.worker.clone()).collect()
    }

    /// List available workers (online + ready status)
    ///
    /// Returns workers that are:
    /// 1. Online (recent heartbeat)
    /// 2. Ready status (not busy/starting/stopped)
    pub fn list_available_workers(&self) -> Vec<WorkerInfo> {
        let workers = self.workers.read().unwrap();
        workers
            .values()
            .filter(|hb| hb.is_recent() && hb.worker.is_available())
            .map(|hb| hb.worker.clone())
            .collect()
    }

    /// Find workers serving a specific model
    pub fn find_workers_by_model(&self, model_id: &str) -> Vec<WorkerInfo> {
        let workers = self.workers.read().unwrap();
        workers
            .values()
            .filter(|hb| hb.is_recent() && hb.worker.serves_model(model_id))
            .map(|hb| hb.worker.clone())
            .collect()
    }

    /// Find best worker for model
    ///
    /// Finds the best available worker for a given model:
    /// 1. Must be online (recent heartbeat)
    /// 2. Must be available (Ready status)
    /// 3. Must serve the requested model
    /// 4. Prefers worker with lowest load (future: use metrics)
    pub fn find_best_worker_for_model(&self, model_id: &str) -> Option<WorkerInfo> {
        let workers = self.workers.read().unwrap();
        workers
            .values()
            .filter(|hb| {
                hb.is_recent() && hb.worker.is_available() && hb.worker.serves_model(model_id)
            })
            .map(|hb| hb.worker.clone())
            .next() // For now, just return first match
                    // Future: Sort by load/metrics
    }

    /// Get total worker count (including stale)
    pub fn total_worker_count(&self) -> usize {
        let workers = self.workers.read().unwrap();
        workers.len()
    }

    /// Get online worker count
    pub fn online_worker_count(&self) -> usize {
        let workers = self.workers.read().unwrap();
        workers.values().filter(|hb| hb.is_recent()).count()
    }

    /// Check if worker is online
    pub fn is_worker_online(&self, worker_id: &str) -> bool {
        let workers = self.workers.read().unwrap();
        workers.get(worker_id).map(|hb| hb.is_recent()).unwrap_or(false)
    }

    /// Clean up stale workers
    ///
    /// Removes workers that haven't sent heartbeat in timeout window.
    /// Returns number of workers removed.
    pub fn cleanup_stale_workers(&self) -> usize {
        let mut workers = self.workers.write().unwrap();
        let initial_count = workers.len();

        workers.retain(|_, hb| hb.is_recent());

        initial_count - workers.len()
    }
}

impl Default for WorkerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use worker_contract::WorkerStatus;

    fn create_worker(id: &str, model: &str, status: WorkerStatus) -> WorkerInfo {
        WorkerInfo {
            id: id.to_string(),
            model_id: model.to_string(),
            device: "GPU-0".to_string(),
            port: 9301,
            status,
            implementation: "llm-worker-rbee".to_string(),
            version: "0.1.0".to_string(),
        }
    }

    #[test]
    fn test_update_and_get_worker() {
        let registry = WorkerRegistry::new();
        let worker = create_worker("worker-1", "llama-3-8b", WorkerStatus::Ready);
        let heartbeat = WorkerHeartbeat::new(worker.clone());

        registry.update_worker(heartbeat);

        let retrieved = registry.get_worker("worker-1").unwrap();
        assert_eq!(retrieved.id, "worker-1");
        assert_eq!(retrieved.model_id, "llama-3-8b");
    }

    #[test]
    fn test_remove_worker() {
        let registry = WorkerRegistry::new();
        let worker = create_worker("worker-1", "llama-3-8b", WorkerStatus::Ready);
        registry.update_worker(WorkerHeartbeat::new(worker));

        assert_eq!(registry.total_worker_count(), 1);

        let removed = registry.remove_worker("worker-1");
        assert!(removed);
        assert_eq!(registry.total_worker_count(), 0);
    }

    #[test]
    fn test_list_available_workers() {
        let registry = WorkerRegistry::new();

        // Add ready worker
        let worker1 = create_worker("worker-1", "llama-3-8b", WorkerStatus::Ready);
        registry.update_worker(WorkerHeartbeat::new(worker1));

        // Add busy worker
        let worker2 = create_worker("worker-2", "llama-3-8b", WorkerStatus::Busy);
        registry.update_worker(WorkerHeartbeat::new(worker2));

        let available = registry.list_available_workers();
        assert_eq!(available.len(), 1);
        assert_eq!(available[0].id, "worker-1");
    }

    #[test]
    fn test_find_workers_by_model() {
        let registry = WorkerRegistry::new();

        let worker1 = create_worker("worker-1", "llama-3-8b", WorkerStatus::Ready);
        registry.update_worker(WorkerHeartbeat::new(worker1));

        let worker2 = create_worker("worker-2", "mistral-7b", WorkerStatus::Ready);
        registry.update_worker(WorkerHeartbeat::new(worker2));

        let llama_workers = registry.find_workers_by_model("llama-3-8b");
        assert_eq!(llama_workers.len(), 1);
        assert_eq!(llama_workers[0].id, "worker-1");
    }

    #[test]
    fn test_find_best_worker_for_model() {
        let registry = WorkerRegistry::new();

        let worker1 = create_worker("worker-1", "llama-3-8b", WorkerStatus::Ready);
        registry.update_worker(WorkerHeartbeat::new(worker1));

        let worker2 = create_worker("worker-2", "llama-3-8b", WorkerStatus::Busy);
        registry.update_worker(WorkerHeartbeat::new(worker2));

        let best = registry.find_best_worker_for_model("llama-3-8b").unwrap();
        assert_eq!(best.id, "worker-1"); // Only ready worker
    }

    #[test]
    fn test_online_worker_count() {
        let registry = WorkerRegistry::new();

        let worker1 = create_worker("worker-1", "llama-3-8b", WorkerStatus::Ready);
        registry.update_worker(WorkerHeartbeat::new(worker1));

        let worker2 = create_worker("worker-2", "mistral-7b", WorkerStatus::Ready);
        registry.update_worker(WorkerHeartbeat::new(worker2));

        assert_eq!(registry.online_worker_count(), 2);
        assert_eq!(registry.total_worker_count(), 2);
    }

    #[test]
    fn test_is_worker_online() {
        let registry = WorkerRegistry::new();

        let worker = create_worker("worker-1", "llama-3-8b", WorkerStatus::Ready);
        registry.update_worker(WorkerHeartbeat::new(worker));

        assert!(registry.is_worker_online("worker-1"));
        assert!(!registry.is_worker_online("nonexistent"));
    }

    #[test]
    fn test_cleanup_stale_workers() {
        let registry = WorkerRegistry::new();

        // Add worker with old heartbeat
        let worker = create_worker("worker-1", "llama-3-8b", WorkerStatus::Ready);
        let old_timestamp = Utc::now()
            - chrono::Duration::seconds(worker_contract::HEARTBEAT_TIMEOUT_SECS as i64 + 10);
        let old_heartbeat = WorkerHeartbeat { worker, timestamp: old_timestamp };
        registry.update_worker(old_heartbeat);

        assert_eq!(registry.total_worker_count(), 1);

        let removed = registry.cleanup_stale_workers();
        assert_eq!(removed, 1);
        assert_eq!(registry.total_worker_count(), 0);
    }
}
