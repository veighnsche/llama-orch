//! TGI HTTP adapter (stub-only, no network calls).

use contracts_api_types as api;
use worker_adapters_adapter_api::{
    TokenEvent, TokenStream, WorkerAdapter, WorkerError, WorkerHealth, WorkerProps,
};

#[derive(Debug, Clone)]
pub struct TgiHttpAdapter {
    pub base_url: String,
}

impl TgiHttpAdapter {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
        }
    }
}

impl WorkerAdapter for TgiHttpAdapter {
    fn health(&self) -> Result<WorkerHealth, WorkerError> {
        unimplemented!()
    }
    fn props(&self) -> Result<WorkerProps, WorkerError> {
        unimplemented!()
    }
    fn submit(&self, _req: api::TaskRequest) -> Result<TokenStream, WorkerError> {
        unimplemented!()
    }
    fn cancel(&self, _task_id: &str) -> Result<(), WorkerError> {
        unimplemented!()
    }
    fn engine_version(&self) -> Result<String, WorkerError> {
        unimplemented!()
    }
}
