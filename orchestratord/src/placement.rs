//! Placement policy hooks (planning-only).

use std::sync::Arc;

use contracts_api_types as api;
use worker_adapters_adapter_api::WorkerAdapter;

use crate::state::AppState;

/// Minimal adapter selection based on requested engine.
/// In the planning phase, this maps all engines to a single mock adapter.
pub fn choose_adapter<'a>(state: &'a AppState, engine: &api::Engine) -> Option<Arc<dyn WorkerAdapter>> {
    let key = match engine {
        api::Engine::Llamacpp => "llamacpp",
        api::Engine::Vllm => "vllm",
        api::Engine::Tgi => "tgi",
        api::Engine::Triton => "triton",
    };
    let map = state.adapters.lock().ok()?;
    map.get(key).cloned()
}
