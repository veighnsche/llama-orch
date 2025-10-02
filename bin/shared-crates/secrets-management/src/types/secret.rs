//! Secret type for string-based secrets (API tokens, passwords)
//!
//! Wraps sensitive string data with automatic zeroization on drop and
//! timing-safe verification.

use secrecy::{ExposeSecret, Secret as SecrecySecret};
use subtle::ConstantTimeEq;
use zeroize::Zeroizing;

/// Opaque wrapper for string-based secrets (API tokens, passwords)
///
/// # Security Properties
///
/// - No Debug/Display implementation (prevents accidental logging)
/// - Automatic zeroization on drop
/// - Timing-safe verification
///
/// # Example
///
/// ```rust,no_run
/// use secrets_management::Secret;
///
/// # fn main() -> Result<(), secrets_management::SecretError> {
/// let token = Secret::load_from_file("/etc/llorch/secrets/api-token")?;
///
/// // Verify incoming request (timing-safe)
/// let received_token = "user-provided-token";
/// if token.verify(received_token) {
///     println!("Authenticated");
/// }
///
/// // Expose for outbound requests (use sparingly)
/// let auth_header = format!("Bearer {}", token.expose());
/// # Ok(())
/// # }
/// ```
pub struct Secret {
    inner: SecrecySecret<Zeroizing<String>>,
}

impl Secret {
    /// Create a new secret from a string
    ///
    /// # Security
    ///
    /// The input string will be zeroized when this Secret is dropped.
    pub(crate) fn new(value: String) -> Self {
        Self { inner: SecrecySecret::new(Zeroizing::new(value)) }
    }

    /// Verify input matches secret using constant-time comparison
    ///
    /// # Security
    ///
    /// Uses `subtle::ConstantTimeEq` to prevent timing attacks (CWE-208).
    /// Examines all bytes regardless of match status.
    /// Length comparison can short-circuit (length is public information).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use secrets_management::Secret;
    /// # fn main() -> Result<(), secrets_management::SecretError> {
    /// # let token = Secret::load_from_file("/tmp/token")?;
    /// let received_token = "user-provided-token";
    /// if token.verify(received_token) {
    ///     // Authenticated
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn verify(&self, input: &str) -> bool {
        let secret_value = self.inner.expose_secret();

        // Length check can short-circuit (length is not secret)
        if secret_value.len() != input.len() {
            return false;
        }

        // Constant-time comparison of bytes using subtle crate
        // This prevents timing attacks by examining all bytes
        secret_value.as_bytes().ct_eq(input.as_bytes()).into()
    }

    /// Expose the secret value (use sparingly)
    ///
    /// # Security Warning
    ///
    /// This exposes the raw secret value. Only use when necessary
    /// (e.g., constructing Bearer token headers).
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use secrets_management::Secret;
    /// # let token = Secret::load_from_file("/tmp/token")?;
    /// let auth_header = format!("Bearer {}", token.expose());
    /// # Ok::<(), secrets_management::SecretError>(())
    /// ```
    #[must_use]
    pub fn expose(&self) -> &str {
        self.inner.expose_secret()
    }
}

// No Debug/Display/ToString/Serialize implementations
// (prevents accidental logging of secrets)

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_matching() {
        let secret = Secret::new("test-secret".to_string());
        assert!(secret.verify("test-secret"));
    }

    #[test]
    fn test_verify_non_matching() {
        let secret = Secret::new("test-secret".to_string());
        assert!(!secret.verify("wrong-secret"));
    }

    #[test]
    fn test_verify_length_mismatch() {
        let secret = Secret::new("short".to_string());
        assert!(!secret.verify("much-longer-token"));
    }

    #[test]
    fn test_expose() {
        let secret = Secret::new("test-secret".to_string());
        assert_eq!(secret.expose(), "test-secret");
    }
}
