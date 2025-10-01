//! adapter-host — in-process registry and facade for WorkerAdapter implementations.

use contracts_api_types as types;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use worker_adapters_adapter_api as api;

pub type PoolId = String;
pub type ReplicaId = String;
type AdapterKey = (PoolId, ReplicaId);
type AdapterRef = Arc<dyn api::WorkerAdapter>;
type Registry = HashMap<AdapterKey, AdapterRef>;

#[derive(Default, Clone)]
pub struct AdapterHost {
    registry: Arc<RwLock<Registry>>,
}

impl AdapterHost {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn bind(&self, pool: PoolId, replica: ReplicaId, adapter: Arc<dyn api::WorkerAdapter>) {
        self.registry.write().unwrap().insert((pool, replica), adapter);
    }

    pub fn submit(&self, pool: &str, req: types::TaskRequest) -> anyhow::Result<api::TokenStream> {
        // Minimal stub: locate any replica for the pool and forward. Real impl adds retries/breakers.
        let guard = self.registry.read().unwrap();
        if let Some((_, adapter)) = guard.iter().find(|((p, _), _)| p == pool) {
            adapter.submit(req).map_err(|e| anyhow::anyhow!(e.to_string()))
        } else {
            Err(anyhow::anyhow!("no adapter bound for pool"))
        }
    }

    pub fn cancel(&self, pool: &str, task_id: &str) -> anyhow::Result<()> {
        let guard = self.registry.read().unwrap();
        if let Some((_, adapter)) = guard.iter().find(|((p, _), _)| p == pool) {
            adapter.cancel(task_id).map_err(|e| anyhow::anyhow!(e.to_string()))
        } else {
            Err(anyhow::anyhow!("no adapter bound for pool"))
        }
    }
}
