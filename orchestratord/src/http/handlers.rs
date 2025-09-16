// Compatibility shim for legacy imports in BDD harness and other callers.
// Re-export the modular handlers under the old module path.

pub use super::catalog::{create_catalog_model, get_catalog_model, verify_catalog_model};
pub use super::control::{
    drain_pool, get_capabilities, get_pool_health, list_replicasets, reload_pool, set_model_state,
};
pub use super::data::{cancel_task, create_task, delete_session, get_session, stream_task};
pub use super::observability::metrics_endpoint;
