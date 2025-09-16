use std::{collections::HashMap, sync::{Arc, Mutex}};
use worker_adapters_adapter_api::WorkerAdapter;

use orchestrator_core::queue::Policy;

use crate::admission::{MetricLabels, QueueWithMetrics};

#[derive(Clone)]
pub struct AppState {
    pub queue: Arc<Mutex<QueueWithMetrics>>,
    pub model_state: Arc<Mutex<ModelState>>,
    pub logs: Arc<Mutex<Vec<String>>>,
    pub pools: Arc<Mutex<HashMap<String, PoolHealth>>>,
    pub adapters: Arc<Mutex<HashMap<String, Arc<dyn WorkerAdapter>>>>,
    pub sse: Arc<Mutex<HashMap<String, String>>>,
}

#[derive(Clone, Debug)]
pub enum ModelState {
    Draft,
    Deprecated { deadline_ms: i64 },
    Retired,
}

#[derive(Clone, Debug)]
pub struct PoolHealth {
    pub live: bool,
    pub ready: bool,
    pub draining: bool,
    pub metrics: serde_json::Value,
}

pub fn default_state() -> AppState {
    // Initialize default labels for a placeholder pool/replica
    let labels = MetricLabels {
        engine: "llamacpp".to_string(),
        engine_version: "v0".to_string(),
        pool_id: "pool0".to_string(),
        replica_id: "r0".to_string(),
    };
    let queue = QueueWithMetrics::new(1024, Policy::DropLru, labels);
    AppState {
        queue: Arc::new(Mutex::new(queue)),
        model_state: Arc::new(Mutex::new(ModelState::Draft)),
        logs: Arc::new(Mutex::new(Vec::new())),
        pools: {
            let mut m = HashMap::new();
            m.insert("pool0".to_string(), PoolHealth { live: true, ready: false, draining: false, metrics: serde_json::json!({"queue_depth":0}) });
            Arc::new(Mutex::new(m))
        },
        adapters: {
            let mut m: HashMap<String, Arc<dyn WorkerAdapter>> = HashMap::new();
            // Map all engines to a mock adapter for the vertical slice
            let mock = Arc::new(worker_adapters_mock::MockAdapter::default()) as Arc<dyn WorkerAdapter>;
            m.insert("llamacpp".to_string(), mock.clone());
            m.insert("vllm".to_string(), mock.clone());
            m.insert("tgi".to_string(), mock.clone());
            m.insert("triton".to_string(), mock);
            Arc::new(Mutex::new(m))
        },
        sse: Arc::new(Mutex::new(HashMap::new())),
    }
}

impl std::fmt::Debug for AppState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let logs_len = self.logs.lock().map(|l| l.len()).unwrap_or_default();
        let pools_len = self.pools.lock().map(|p| p.len()).unwrap_or_default();
        let adapters_len = self.adapters.lock().map(|a| a.len()).unwrap_or_default();
        let sse_len = self.sse.lock().map(|m| m.len()).unwrap_or_default();
        f.debug_struct("AppState")
            .field("queue", &"<QueueWithMetrics>")
            .field("model_state", &"<ModelState>")
            .field("logs_len", &logs_len)
            .field("pools_len", &pools_len)
            .field("adapters_len", &adapters_len)
            .field("sse_len", &sse_len)
            .finish()
    }
}
