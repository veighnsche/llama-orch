use pool_managerd::core::registry::Registry;
use provisioners_engine_provisioner::PreparedEngine;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Debug, cucumber::World)]
pub struct BddWorld {
    /// Shared registry for testing
    pub registry: Arc<Mutex<Registry>>,

    /// HTTP client for API calls
    pub client: Option<reqwest::Client>,

    /// Base URL for daemon
    pub base_url: String,

    /// Last HTTP response
    pub last_status: Option<u16>,
    pub last_headers: Option<reqwest::header::HeaderMap>,
    pub last_body: Option<String>,

    /// PreparedEngine for current scenario
    pub prepared_engine: Option<PreparedEngine>,

    /// Test fixtures
    pub pool_id: Option<String>,
    pub handoff_json: Option<serde_json::Value>,

    /// Process tracking
    pub spawned_pids: Vec<u32>,
    pub pid_files: Vec<PathBuf>,

    /// Mock behaviors
    pub mock_health_responses: HashMap<String, bool>, // pool_id -> will_respond
    pub mock_health_delay_ms: HashMap<String, u64>, // pool_id -> delay

    /// Assertions
    pub facts: Vec<serde_json::Value>,
}

impl Default for BddWorld {
    fn default() -> Self {
        Self {
            registry: Arc::new(Mutex::new(Registry::new())),
            client: Some(reqwest::Client::new()),
            base_url: "http://127.0.0.1:9200".to_string(),
            last_status: None,
            last_headers: None,
            last_body: None,
            prepared_engine: None,
            pool_id: None,
            handoff_json: None,
            spawned_pids: Vec::new(),
            pid_files: Vec::new(),
            mock_health_responses: HashMap::new(),
            mock_health_delay_ms: HashMap::new(),
            facts: Vec::new(),
        }
    }
}

impl BddWorld {
    pub fn push_fact<S: AsRef<str>>(&mut self, stage: S) {
        self.facts.push(serde_json::json!({
            "stage": stage.as_ref(),
        }));
    }
}
