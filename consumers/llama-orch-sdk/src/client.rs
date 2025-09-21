// Design-phase stub client — signatures only; all methods unimplemented.
// Mirrors consumers/llama-orch-sdk/.docs/03-client.md

use crate::types::{
    AdmissionResponse, EngineCapability, SSEEvent, SessionInfo, TaskRequest,
};

#[derive(Debug, Clone)]
pub struct ClientOptions {
    pub base_url: Option<String>,
    pub api_key: Option<String>, // sent as X-API-Key when provided
    pub timeout_ms: Option<u64>, // non-streaming calls
}

#[derive(Debug, Clone)]
pub struct OrchestratorClient {
    base_url: String,
    api_key: Option<String>,
    timeout_ms: u64,
}

impl Default for OrchestratorClient {
    fn default() -> Self {
        Self::new(ClientOptions {
            base_url: None,
            api_key: None,
            timeout_ms: None,
        })
    }
}

impl OrchestratorClient {
    pub fn new(opts: ClientOptions) -> Self {
        Self {
            base_url: opts.base_url.unwrap_or_else(|| "http://127.0.0.1:8080/".to_string()),
            api_key: opts.api_key,
            timeout_ms: opts.timeout_ms.unwrap_or(30_000),
        }
    }

    pub async fn list_engines(&self) -> Result<Vec<EngineCapability>, String> {
        let _ = &self.base_url;
        Err("unimplemented".to_string())
    }

    pub async fn enqueue_task(&self, _req: TaskRequest) -> Result<AdmissionResponse, String> {
        Err("unimplemented".to_string())
    }

    pub async fn stream_task(
        &self,
        _task_id: &str,
    ) -> Result<std::pin::Pin<Box<dyn futures_core::Stream<Item = SSEEvent> + 'static>>, String> {
        Err("unimplemented".to_string())
    }

    pub async fn cancel_task(&self, _task_id: &str) -> Result<(), String> {
        Err("unimplemented".to_string())
    }

    pub async fn get_session(&self, _session_id: &str) -> Result<SessionInfo, String> {
        Err("unimplemented".to_string())
    }
}
