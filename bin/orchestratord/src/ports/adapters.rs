#[derive(Debug, Clone)]
pub struct AdapterProps {
    pub ctx_max: i32,
    pub supported_workloads: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct StreamRequest {
    pub prompt: String,
}
// TODO(ARCH-CHANGE): This entire file will be removed per ARCHITECTURE_CHANGE_PLAN.md Phase 2.
// The adapter abstraction is being replaced with direct worker-orcd communication.
// See: .docs/ARCHITECTURE_CHANGE_PLAN.md §2 (Components to Remove)
#[derive(Debug)]
pub struct AdapterStream; // placeholder - REMOVE with adapter-host

#[derive(Debug, Clone)]
pub enum StreamItem {
    Started { queue_position: i32, predicted_start_ms: i64 },
    Token { t: String, i: i32 },
    Metrics { json: serde_json::Value },
    End { tokens_out: i32, decode_ms: i32 },
    Error { message: String },
}

pub trait AdapterClient: Send + Sync {
    fn stream(&self, req: &StreamRequest) -> Vec<StreamItem>;
    fn props(&self) -> anyhow::Result<AdapterProps>;
}

pub trait AdapterRegistry: Send + Sync {
    fn engines(&self) -> Vec<String>;
    fn props(&self, engine: &str) -> anyhow::Result<AdapterProps>;
    fn client(&self, engine: &str) -> anyhow::Result<std::sync::Arc<dyn AdapterClient>>;
}
