//! model-provisioner — orchestrates model ensure-present flows using catalog-core.
//!
//! Responsibilities:
//! - Parse ModelRef input (string or structured) and ensure the model is present locally.
//! - Verify digests when given; warn otherwise (per spec §2.6/§2.11).
//! - Register/update catalog entries and lifecycle state.
//! - Return ResolvedModel with canonical local path for engine-provisioner and pool-managerd.
// Module structure
pub mod api;
pub mod config;
pub mod metadata;
pub mod provisioner;
mod util;

// Re-exports for crate public API
pub use api::{
    provision_from_config_to_default_handoff, provision_from_config_to_handoff,
    DEFAULT_LLAMACPP_HANDOFF_PATH,
};
pub use config::ModelProvisionerConfig;
pub use metadata::ModelMetadata;
pub use provisioner::ModelProvisioner;

// Shared test locks to serialize global env/CWD mutations across modules
#[cfg(test)]
pub mod test_locks {
    use std::sync::{Mutex, OnceLock};
    pub static CWD_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    pub static PATH_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
}
