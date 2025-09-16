//! Mock adapter (stub-only, no I/O).

use contracts_api_types as api;
use worker_adapters_adapter_api::{
    TokenEvent, TokenStream, WorkerAdapter, WorkerError, WorkerHealth, WorkerProps,
};
use futures::{stream, StreamExt};

#[derive(Debug, Clone, Default)]
pub struct MockAdapter;

impl WorkerAdapter for MockAdapter {
    fn health(&self) -> Result<WorkerHealth, WorkerError> {
        Ok(WorkerHealth { live: true, ready: true })
    }
    fn props(&self) -> Result<WorkerProps, WorkerError> {
        Ok(WorkerProps { slots_total: Some(1), slots_free: Some(1) })
    }
    fn submit(&self, _req: api::TaskRequest) -> Result<TokenStream, WorkerError> {
        let events = vec![
            Ok(TokenEvent { kind: "started".into(), data: serde_json::json!({}) }),
            Ok(TokenEvent { kind: "token".into(), data: serde_json::json!({"text":"hello"}) }),
            Ok(TokenEvent { kind: "end".into(), data: serde_json::json!({}) }),
        ];
        Ok(stream::iter(events).boxed())
    }
    fn cancel(&self, _task_id: &str) -> Result<(), WorkerError> {
        Ok(())
    }
    fn engine_version(&self) -> Result<String, WorkerError> {
        Ok("mock-stub-v0".into())
    }
}
