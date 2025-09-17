// Compatibility shim for legacy imports in BDD harness and other callers.
// Re-export the modular handlers under the old module path.

// SHOULD BE REMOVED BECAUSE IT IS NOT NEEDED ANYMORE
// THE FIRST GOLDEN RULE IS THAT NO BACKWARDS COMPATIBILITY IS NEEDED PRE-v1.0.0

pub use super::catalog::{create_catalog_model, get_catalog_model, verify_catalog_model};
pub use super::control::{
    drain_pool, get_capabilities, get_pool_health, list_replicasets, reload_pool, set_model_state,
};
pub use super::data::{cancel_task, create_task, delete_session, get_session, stream_task};
pub use super::observability::metrics_endpoint;
