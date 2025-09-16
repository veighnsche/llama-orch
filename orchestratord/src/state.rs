use std::sync::{Arc, Mutex};

use orchestrator_core::queue::Policy;

use crate::admission::{MetricLabels, QueueWithMetrics};

#[derive(Clone)]
pub struct AppState {
    pub queue: Arc<Mutex<QueueWithMetrics>>,
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
    }
}
