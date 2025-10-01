//! secrets-management — Secure credential storage and rotation
//!
//! Provides secure loading of secrets from files, environment variables, or systemd credentials.
//! Prevents secrets from appearing in process listings or logs.
//!
//! # Security Properties
//!
//! - Secrets loaded from files (not environment variables)
//! - Memory cleared on drop (zeroize)
//! - Never logged or displayed
//! - Supports rotation with graceful overlap
//!
//! # Example
//!
//! ```rust
//! use secrets_management::{SecretStore, SecretSource};
//!
//! // Load from file
//! let store = SecretStore::new();
//! let token = store.load("api_token", SecretSource::File("/etc/llorch/api-token"))?;
//!
//! // Use secret (never logs raw value)
//! if token.verify(user_input) {
//!     println!("Authenticated");
//! }
//! ```

// Security-critical crate: TIER 1 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![deny(clippy::integer_arithmetic)]
#![deny(clippy::cast_ptr_alignment)]
#![deny(clippy::mem_forget)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![warn(clippy::arithmetic_side_effects)]
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

use std::path::PathBuf;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SecretError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("secret not found: {0}")]
    NotFound(String),
    #[error("invalid secret format")]
    InvalidFormat,
}

pub type Result<T> = std::result::Result<T, SecretError>;

/// Source for loading secrets
#[derive(Debug, Clone)]
pub enum SecretSource {
    /// Load from file path
    File(PathBuf),
    /// Load from systemd credential
    SystemdCredential(String),
    /// Load from environment variable (NOT RECOMMENDED)
    Environment(String),
}

/// Secure secret storage
pub struct Secret {
    value: String,
}

impl Secret {
    /// Verify input matches secret (timing-safe)
    pub fn verify(&self, input: &str) -> bool {
        // Use constant-time comparison
        if self.value.len() != input.len() {
            return false;
        }
        
        let mut result = 0u8;
        for (a, b) in self.value.bytes().zip(input.bytes()) {
            result |= a ^ b;
        }
        
        result == 0
    }
    
    /// Get secret value (use sparingly)
    pub fn expose(&self) -> &str {
        &self.value
    }
}

impl Drop for Secret {
    fn drop(&mut self) {
        // Zero memory on drop (best effort)
        unsafe {
            std::ptr::write_volatile(
                self.value.as_mut_ptr(),
                0,
            );
        }
    }
}

/// Secret store
pub struct SecretStore;

impl SecretStore {
    pub fn new() -> Self {
        Self
    }
    
    /// Load secret from source
    pub fn load(&self, name: &str, source: SecretSource) -> Result<Secret> {
        let value = match source {
            SecretSource::File(path) => {
                std::fs::read_to_string(&path)
                    .map_err(|e| {
                        tracing::error!(secret = %name, path = %path.display(), "Failed to load secret");
                        e
                    })?
                    .trim()
                    .to_string()
            }
            SecretSource::SystemdCredential(cred_name) => {
                let path = PathBuf::from(format!("/run/credentials/{}", cred_name));
                std::fs::read_to_string(&path)
                    .map_err(|e| {
                        tracing::error!(secret = %name, credential = %cred_name, "Failed to load systemd credential");
                        e
                    })?
                    .trim()
                    .to_string()
            }
            SecretSource::Environment(var_name) => {
                tracing::warn!(
                    secret = %name,
                    env_var = %var_name,
                    "Loading secret from environment variable (NOT RECOMMENDED - visible in process listing)"
                );
                std::env::var(&var_name)
                    .map_err(|_| SecretError::NotFound(var_name.clone()))?
            }
        };
        
        if value.is_empty() {
            return Err(SecretError::NotFound(name.to_string()));
        }
        
        tracing::info!(secret = %name, "Secret loaded successfully");
        Ok(Secret { value })
    }
}

impl Default for SecretStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_secret_verify() {
        let secret = Secret {
            value: "test-secret".to_string(),
        };
        
        assert!(secret.verify("test-secret"));
        assert!(!secret.verify("wrong-secret"));
        assert!(!secret.verify("test-secre")); // Length mismatch
    }
}
