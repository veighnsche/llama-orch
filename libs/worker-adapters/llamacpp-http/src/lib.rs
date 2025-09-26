//! llama.cpp HTTP adapter (stub-only, no network calls).

use contracts_api_types as api;
use futures::{stream, StreamExt};
use worker_adapters_adapter_api::{
    TokenEvent, TokenStream, WorkerAdapter, WorkerError, WorkerHealth, WorkerProps,
};

#[derive(Debug, Clone)]
pub struct LlamaCppHttpAdapter {
    pub base_url: String,
}

impl LlamaCppHttpAdapter {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
        }
    }
}

impl WorkerAdapter for LlamaCppHttpAdapter {
    fn health(&self) -> Result<WorkerHealth, WorkerError> {
        Ok(WorkerHealth {
            live: true,
            ready: true,
        })
    }

    fn props(&self) -> Result<WorkerProps, WorkerError> {
        Ok(WorkerProps {
            slots_total: Some(1),
            slots_free: Some(1),
        })
    }

    fn submit(&self, _req: api::TaskRequest) -> Result<TokenStream, WorkerError> {
        let events = vec![
            Ok(TokenEvent {
                kind: "started".into(),
                data: serde_json::json!({}),
            }),
            Ok(TokenEvent {
                kind: "token".into(),
                data: serde_json::json!({"t":"hello","i":0}),
            }),
            Ok(TokenEvent {
                kind: "end".into(),
                data: serde_json::json!({"tokens_out":1,"decode_ms":0}),
            }),
        ];
        Ok(stream::iter(events).boxed())
    }

    fn cancel(&self, _task_id: &str) -> Result<(), WorkerError> {
        Ok(())
    }

    fn engine_version(&self) -> Result<String, WorkerError> {
        Ok("llamacpp-stub-v0".into())
    }
}
