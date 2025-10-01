//! model-provisioner — orchestrates model ensure-present flows using catalog-core.
//!
//! Responsibilities:
//! - Parse ModelRef input (string or structured) and ensure the model is present locally.
//! - Verify digests when given; warn otherwise (per spec §2.6/§2.11).
//! - Register/update catalog entries and lifecycle state.
//! - Return ResolvedModel with canonical local path for engine-provisioner and pool-managerd.
//!
//! # ⚠️ INPUT VALIDATION REMINDER
//!
//! **CRITICAL**: Always validate model references with `input-validation` crate:
//!
//! ```rust,ignore
//! use input_validation::{
//!     validate_model_ref,     // For model references (HuggingFace, file paths, URLs)
//!     validate_path,          // For local file paths
//!     validate_hex_string,    // For SHA-256 digests
//!     validate_identifier,    // For model IDs, catalog keys
//! };
//!
//! // Example: Validate model reference before provisioning
//! validate_model_ref(&model_ref)?;  // Prevents command injection in wget/curl/git
//!
//! // Example: Validate digest
//! validate_hex_string(&digest, 64)?;  // SHA-256 = 64 hex chars
//!
//! // Example: Validate local path
//! validate_path(&local_path, &allowed_models_dir)?;  // Prevents path traversal
//! ```
//!
//! **Why?** Model provisioning is HIGH RISK for command injection:
//! - ✅ Shell metacharacter blocking (`;`, `|`, `&`, `$`, `` ` ``)
//! - ✅ Path traversal prevention (`../`, `..\`)
//! - ✅ Command substitution prevention
//! - ✅ Real-world pattern support (HuggingFace, file:, https:)
//!
//! See: `bin/shared-crates/input-validation/README.md`
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
