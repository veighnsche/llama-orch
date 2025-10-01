//! Environment variable loading (DEPRECATED)
//!
//! Loads secrets from environment variables with security warnings.
//!
//! # Security Warning
//!
//! Environment variables are visible in:
//! - Process listings (`ps auxe`)
//! - `/proc/PID/environ`
//! - Docker inspect output
//! - Systemd service files
//!
//! **Use file-based loading instead.**

use crate::{Secret, Result, SecretError};

/// Load a secret from environment variable (NOT RECOMMENDED)
///
/// # Security Warning
///
/// Environment variables are visible in process listings and `/proc`.
/// Use [`Secret::load_from_file`] instead.
///
/// # Example
///
/// ```rust,no_run
/// use secrets_management::Secret;
///
/// // ❌ NOT RECOMMENDED: Visible in process listing
/// let token = Secret::from_env("LLORCH_API_TOKEN")?;
/// // ✅ RECOMMENDED: Use file-based loading
/// let token = Secret::load_from_file("/etc/llorch/secrets/api-token")?;
/// # Ok::<(), secrets_management::SecretError>(())
/// ```
#[deprecated(
    since = "0.1.0",
    note = "Environment variables are visible in process listings. Use Secret::load_from_file instead."
)]
pub fn load_from_env(var_name: &str) -> Result<Secret> {
    tracing::warn!(
        env_var = %var_name,
        "Loading secret from environment variable (NOT RECOMMENDED - visible in process listing and /proc)"
    );
    
    let value = std::env::var(var_name)
        .map_err(|_| SecretError::FileNotFound(format!("env var not set: {}", var_name)))?;
    
    if value.is_empty() {
        return Err(SecretError::InvalidFormat("empty value".to_string()));
    }
    
    Ok(Secret::new(value))
}

impl Secret {
    /// Load from environment variable (NOT RECOMMENDED)
    ///
    /// See [`load_from_env`] for details and security warnings.
    ///
    /// # Errors
    ///
    /// Returns `SecretError` if environment variable not set or empty.
    #[deprecated(
        since = "0.1.0",
        note = "Environment variables are visible in process listings. Use Secret::load_from_file instead."
    )]
    #[allow(deprecated)]
    pub fn from_env(var_name: &str) -> Result<Self> {
        load_from_env(var_name)
    }
}

#[cfg(test)]
mod tests {
    use super::load_from_env;
    
    #[test]
    #[allow(deprecated)]
    fn test_load_from_env() {
        std::env::set_var("TEST_SECRET", "test-value");
        let secret = load_from_env("TEST_SECRET").unwrap();
        assert_eq!(secret.expose(), "test-value");
        std::env::remove_var("TEST_SECRET");
    }
    
    #[test]
    #[allow(deprecated)]
    fn test_reject_empty_env() {
        std::env::set_var("TEST_SECRET", "");
        let result = load_from_env("TEST_SECRET");
        assert!(result.is_err());
        std::env::remove_var("TEST_SECRET");
    }
}
