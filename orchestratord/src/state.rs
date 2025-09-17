use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use pool_managerd::registry::Registry as PoolRegistry;

#[derive(Debug, Clone, Default)]
pub struct AppState {
    pub logs: Arc<Mutex<Vec<String>>>,
    pub sessions: Arc<Mutex<HashMap<String, SessionInfo>>>,
    pub artifacts: Arc<Mutex<HashMap<String, serde_json::Value>>>,
    pub pool_manager: Arc<Mutex<PoolRegistry>>,
}

#[derive(Debug, Clone, Default)]
pub struct SessionInfo {
    pub ttl_ms_remaining: i64,
    pub turns: i32,
    pub kv_bytes: i64,
    pub kv_warmth: bool,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            logs: Arc::new(Mutex::new(Vec::new())),
            sessions: Arc::new(Mutex::new(HashMap::new())),
            artifacts: Arc::new(Mutex::new(HashMap::new())),
            pool_manager: Arc::new(Mutex::new(PoolRegistry::new())),
        }
    }
}
