//! Worker registry - in-memory worker tracking
//!
//! Per test-001-mvp.md Phase 5: Worker Startup
//! Tracks spawned workers and their state
//!
//! Created by: TEAM-026

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;
use tokio::sync::RwLock;

/// Worker state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum WorkerState {
    /// Worker is loading the model
    Loading,
    /// Worker is idle and ready for requests
    Idle,
    /// Worker is processing a request
    Busy,
}

/// Worker information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    /// Worker ID (UUID)
    pub id: String,
    /// Worker URL (e.g., "http://mac.home.arpa:8081")
    pub url: String,
    /// Model reference (e.g., "hf:TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF")
    pub model_ref: String,
    /// Backend (e.g., "metal", "cuda", "cpu")
    pub backend: String,
    /// Device ID
    pub device: u32,
    /// Current state
    pub state: WorkerState,
    /// Last activity timestamp (for idle timeout)
    pub last_activity: SystemTime,
    /// Total slots
    pub slots_total: u32,
    /// Available slots
    pub slots_available: u32,
}

/// Worker registry - thread-safe in-memory storage
#[derive(Clone)]
pub struct WorkerRegistry {
    workers: Arc<RwLock<HashMap<String, WorkerInfo>>>,
}

impl WorkerRegistry {
    /// Create new empty registry
    pub fn new() -> Self {
        Self {
            workers: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a new worker
    pub async fn register(&self, worker: WorkerInfo) {
        let mut workers = self.workers.write().await;
        workers.insert(worker.id.clone(), worker);
    }

    /// Update worker state
    pub async fn update_state(&self, worker_id: &str, state: WorkerState) {
        let mut workers = self.workers.write().await;
        if let Some(worker) = workers.get_mut(worker_id) {
            worker.state = state;
            worker.last_activity = SystemTime::now();
        }
    }

    /// Get worker by ID
    pub async fn get(&self, worker_id: &str) -> Option<WorkerInfo> {
        let workers = self.workers.read().await;
        workers.get(worker_id).cloned()
    }

    /// List all workers
    pub async fn list(&self) -> Vec<WorkerInfo> {
        let workers = self.workers.read().await;
        workers.values().cloned().collect()
    }

    /// Remove worker
    pub async fn remove(&self, worker_id: &str) -> Option<WorkerInfo> {
        let mut workers = self.workers.write().await;
        workers.remove(worker_id)
    }

    /// Find idle workers for a specific model
    pub async fn find_idle_worker(&self, model_ref: &str) -> Option<WorkerInfo> {
        let workers = self.workers.read().await;
        workers
            .values()
            .find(|w| w.model_ref == model_ref && w.state == WorkerState::Idle)
            .cloned()
    }

    /// Get idle workers (for timeout enforcement)
    pub async fn get_idle_workers(&self) -> Vec<WorkerInfo> {
        let workers = self.workers.read().await;
        workers
            .values()
            .filter(|w| w.state == WorkerState::Idle)
            .cloned()
            .collect()
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

    #[tokio::test]
    async fn test_registry_register_and_get() {
        let registry = WorkerRegistry::new();

        let worker = WorkerInfo {
            id: "worker-123".to_string(),
            url: "http://localhost:8081".to_string(),
            model_ref: "hf:test/model".to_string(),
            backend: "cpu".to_string(),
            device: 0,
            state: WorkerState::Loading,
            last_activity: SystemTime::now(),
            slots_total: 1,
            slots_available: 1,
        };

        registry.register(worker.clone()).await;

        let retrieved = registry.get("worker-123").await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().id, "worker-123");
    }

    #[tokio::test]
    async fn test_registry_update_state() {
        let registry = WorkerRegistry::new();

        let worker = WorkerInfo {
            id: "worker-123".to_string(),
            url: "http://localhost:8081".to_string(),
            model_ref: "hf:test/model".to_string(),
            backend: "cpu".to_string(),
            device: 0,
            state: WorkerState::Loading,
            last_activity: SystemTime::now(),
            slots_total: 1,
            slots_available: 1,
        };

        registry.register(worker).await;
        registry.update_state("worker-123", WorkerState::Idle).await;

        let retrieved = registry.get("worker-123").await.unwrap();
        assert_eq!(retrieved.state, WorkerState::Idle);
    }

    #[tokio::test]
    async fn test_registry_find_idle_worker() {
        let registry = WorkerRegistry::new();

        let worker1 = WorkerInfo {
            id: "worker-1".to_string(),
            url: "http://localhost:8081".to_string(),
            model_ref: "hf:test/model-a".to_string(),
            backend: "cpu".to_string(),
            device: 0,
            state: WorkerState::Idle,
            last_activity: SystemTime::now(),
            slots_total: 1,
            slots_available: 1,
        };

        let worker2 = WorkerInfo {
            id: "worker-2".to_string(),
            url: "http://localhost:8082".to_string(),
            model_ref: "hf:test/model-b".to_string(),
            backend: "cpu".to_string(),
            device: 0,
            state: WorkerState::Busy,
            last_activity: SystemTime::now(),
            slots_total: 1,
            slots_available: 0,
        };

        registry.register(worker1).await;
        registry.register(worker2).await;

        let found = registry.find_idle_worker("hf:test/model-a").await;
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, "worker-1");

        let not_found = registry.find_idle_worker("hf:test/model-b").await;
        assert!(not_found.is_none()); // worker-2 is busy
    }
}
