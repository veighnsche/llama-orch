// TEAM-109: Audited 2025-10-18 - ✅ CLEAN - In-memory worker registry

//! Worker Registry Module (In-Memory)
//!
//! Created by: TEAM-043
//!
//! Manages ephemeral in-memory registry of active workers.
//! Workers are registered when they start and removed when they stop.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInfo {
    pub id: String,
    pub url: String,
    pub model_ref: String,
    pub backend: String,
    pub device: u32,
    pub state: WorkerState,
    pub slots_total: u32,
    pub slots_available: u32,
    pub vram_bytes: Option<u64>,
    // TEAM-046: Added for worker management
    pub node_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum WorkerState {
    Loading,
    Idle,
    Busy,
}

/// In-memory worker registry
// TEAM-080: Added Clone derive for concurrent testing
#[derive(Clone)]
pub struct WorkerRegistry {
    workers: Arc<RwLock<HashMap<String, WorkerInfo>>>,
}

impl WorkerRegistry {
    /// Create a new empty worker registry
    pub fn new() -> Self {
        Self { workers: Arc::new(RwLock::new(HashMap::new())) }
    }

    /// Register a new worker
    pub async fn register(&self, worker: WorkerInfo) {
        let mut workers = self.workers.write().await;
        workers.insert(worker.id.clone(), worker);
        tracing::info!("Worker registered: {}", workers.len());
    }

    /// Update worker state
    pub async fn update_state(&self, worker_id: &str, state: WorkerState) -> bool {
        let mut workers = self.workers.write().await;
        if let Some(worker) = workers.get_mut(worker_id) {
            worker.state = state;
            true
        } else {
            false
        }
    }

    /// Get a worker by ID
    pub async fn get(&self, worker_id: &str) -> Option<WorkerInfo> {
        let workers = self.workers.read().await;
        workers.get(worker_id).cloned()
    }

    /// List all workers
    pub async fn list(&self) -> Vec<WorkerInfo> {
        let workers = self.workers.read().await;
        workers.values().cloned().collect()
    }

    /// Remove a worker
    pub async fn remove(&self, worker_id: &str) -> bool {
        let mut workers = self.workers.write().await;
        workers.remove(worker_id).is_some()
    }

    /// Clear all workers (for testing)
    pub async fn clear(&self) {
        let mut workers = self.workers.write().await;
        workers.clear();
    }

    /// Count workers
    pub async fn count(&self) -> usize {
        let workers = self.workers.read().await;
        workers.len()
    }

    // TEAM-046: Additional methods for worker management

    /// List all workers with extended info for API
    pub async fn list_workers(&self) -> anyhow::Result<Vec<WorkerInfoExtended>> {
        let workers = self.workers.read().await;
        Ok(workers
            .values()
            .map(|w| WorkerInfoExtended {
                worker_id: w.id.clone(),
                node_name: w.node_name.clone(),
                state: format!("{:?}", w.state).to_lowercase(),
                model_ref: Some(w.model_ref.clone()),
                url: w.url.clone(),
            })
            .collect())
    }

    /// Get workers by node name
    pub async fn get_workers_by_node(&self, node: &str) -> anyhow::Result<Vec<WorkerInfoExtended>> {
        let workers = self.workers.read().await;
        Ok(workers
            .values()
            .filter(|w| w.node_name == node)
            .map(|w| WorkerInfoExtended {
                worker_id: w.id.clone(),
                node_name: w.node_name.clone(),
                state: format!("{:?}", w.state).to_lowercase(),
                model_ref: Some(w.model_ref.clone()),
                url: w.url.clone(),
            })
            .collect())
    }

    /// Shutdown a worker
    pub async fn shutdown_worker(&self, worker_id: &str) -> anyhow::Result<()> {
        let workers = self.workers.read().await;
        if let Some(worker) = workers.get(worker_id) {
            // Send shutdown request to worker
            let client = reqwest::Client::new();
            let response = client.post(format!("{}/v1/admin/shutdown", worker.url)).send().await?;

            if !response.status().is_success() {
                anyhow::bail!("Worker shutdown failed: HTTP {}", response.status());
            }

            Ok(())
        } else {
            anyhow::bail!("Worker not found: {}", worker_id)
        }
    }
}

// TEAM-046: Extended worker info for API responses
#[derive(Debug, Clone, Serialize)]
pub struct WorkerInfoExtended {
    pub worker_id: String,
    pub node_name: String,
    pub state: String,
    pub model_ref: Option<String>,
    pub url: String,
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
    async fn test_worker_registry_crud() {
        let registry = WorkerRegistry::new();

        // Register worker
        let worker = WorkerInfo {
            id: "worker-123".to_string(),
            url: "http://localhost:8081".to_string(),
            model_ref: "tinyllama".to_string(),
            backend: "cuda".to_string(),
            device: 0,
            state: WorkerState::Loading,
            slots_total: 4,
            slots_available: 4,
            vram_bytes: Some(8_000_000_000),
            node_name: "test-node".to_string(),
        };

        registry.register(worker.clone()).await;
        assert_eq!(registry.count().await, 1);

        // Get worker
        let retrieved = registry.get("worker-123").await.unwrap();
        assert_eq!(retrieved.id, "worker-123");
        assert_eq!(retrieved.state, WorkerState::Loading);

        // Update state
        registry.update_state("worker-123", WorkerState::Idle).await;
        let updated = registry.get("worker-123").await.unwrap();
        assert_eq!(updated.state, WorkerState::Idle);

        // List workers
        let workers = registry.list().await;
        assert_eq!(workers.len(), 1);

        // Remove worker
        let removed = registry.remove("worker-123").await;
        assert!(removed);
        assert_eq!(registry.count().await, 0);
    }
}
