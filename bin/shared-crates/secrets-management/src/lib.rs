//! secrets-management — Secure credential storage and management
//!
//! Provides secure loading of API tokens and cryptographic keys from files, systemd credentials,
//! and key derivation. Uses battle-tested libraries (secrecy, zeroize, subtle, hkdf) instead of
//! rolling our own crypto.
//!
//! # Security Properties
//!
//! - **File-based loading** — Load secrets from files (not environment variables)
//! - **Memory safety** — Automatic zeroization on drop (prevents memory dumps)
//! - **Logging safety** — Never logs secret values (only paths/metadata)
//! - **Timing-safe verification** — Constant-time comparison for tokens
//! - **Permission validation** — Rejects world/group-readable files (Unix)
//! - **Key derivation** — HKDF-SHA256 for deriving keys from tokens
//!
//! # Example: Load API Token
//!
//! ```rust,no_run
//! use secrets_management::Secret;
//!
//! # fn main() -> Result<(), secrets_management::SecretError> {
//! // Load token from file
//! let token = Secret::load_from_file("/etc/llorch/secrets/api-token")?;
//!
//! // Verify incoming request (timing-safe)
//! let received_token = "user-provided-token";
//! if token.verify(received_token) {
//!     println!("Authenticated");
//! }
//!
//! // Expose for outbound requests
//! let auth_header = format!("Bearer {}", token.expose());
//! # Ok(())
//! # }
//! ```
//!
//! # Example: Derive Cryptographic Key
//!
//! ```rust,no_run
//! use secrets_management::SecretKey;
//!
//! # fn main() -> Result<(), secrets_management::SecretError> {
//! // Derive seal key from worker token (HKDF-SHA256)
//! let worker_api_token = "worker-token-abc123";
//! let seal_key = SecretKey::derive_from_token(
//!     worker_api_token,
//!     b"llorch-seal-key-v1"  // Domain separation
//! )?;
//!
//! // Use for HMAC-SHA256
//! // Note: This example shows the pattern, actual HMAC usage
//! // would require the hmac and sha2 crates in your Cargo.toml
//! # Ok(())
//! # }
//! ```
//!
//! # Module Organization
//!
//! - [`error`] — Error types and result aliases
//! - [`types`] — Secret types (Secret, SecretKey)
//! - [`loaders`] — Loading methods (file, systemd, derivation)
//! - [`validation`] — File permission and path validation

// Security-critical crate: TIER 1 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![deny(clippy::arithmetic_side_effects)]
#![deny(clippy::cast_ptr_alignment)]
#![deny(clippy::mem_forget)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![warn(clippy::cast_lossless)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::cast_precision_loss)]
#![warn(clippy::cast_sign_loss)]
#![warn(clippy::string_slice)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::must_use_candidate)]

// Module declarations
mod error;
mod types;
mod loaders;
pub mod validation;  // Public for testing and advanced use cases

// Public exports
pub use error::{SecretError, Result};
pub use types::{Secret, SecretKey};

// Re-export commonly used types for convenience
pub use loaders::{
    file::load_secret_from_file,
    file::load_key_from_file,
    systemd::load_from_systemd_credential,
    derivation::derive_key_from_token,
};
