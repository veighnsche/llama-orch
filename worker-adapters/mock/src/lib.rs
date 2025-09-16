//! Mock adapter (stub-only, no I/O).

use contracts_api_types as api;
use worker_adapters_adapter_api::{
    TokenEvent, TokenStream, WorkerAdapter, WorkerError, WorkerHealth, WorkerProps,
};

#[derive(Debug, Clone, Default)]
pub struct MockAdapter;

impl WorkerAdapter for MockAdapter {
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
