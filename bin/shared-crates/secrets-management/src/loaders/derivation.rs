//! Key derivation using HKDF-SHA256
//!
//! Derives cryptographic keys from high-entropy tokens using HKDF-SHA256.
//!
//! # Security
//!
//! - NIST SP 800-108 compliant
//! - RFC 5869 HKDF implementation
//! - Domain separation support
//! - Deterministic (same input → same output)

use hkdf::Hkdf;
use sha2::Sha256;
use crate::{SecretKey, Result, SecretError};

/// Derive a secret key from a token using HKDF-SHA256
///
/// # Security
///
/// - Uses HKDF-SHA256 (NIST SP 800-108 compliant)
/// - Domain separation prevents key reuse across contexts
/// - Deterministic (same token + domain → same key)
/// - Output is 32 bytes (256-bit key)
///
/// # Errors
///
/// Returns `SecretError` if:
/// - Token is empty
/// - Domain separation string is empty
/// - HKDF expansion fails
///
/// # Example
///
/// ```rust,no_run
/// use secrets_management::SecretKey;
///
/// // Derive seal key from worker token
/// let worker_api_token = "worker-token-abc123";
/// let seal_key = SecretKey::derive_from_token(
///     worker_api_token,
///     b"llorch-seal-key-v1"  // Domain separation
/// )?;
///
/// // Different domain produces different key
/// let enc_key = SecretKey::derive_from_token(
///     worker_api_token,
///     b"llorch-encryption-v1"
/// )?;
/// # Ok::<(), secrets_management::SecretError>(())
/// ```
pub fn derive_key_from_token(token: &str, domain: &[u8]) -> Result<SecretKey> {
    if token.is_empty() {
        return Err(SecretError::InvalidFormat("empty token".to_string()));
    }
    
    if domain.is_empty() {
        return Err(SecretError::KeyDerivation(
            "domain separation string required".to_string()
        ));
    }
    
    // HKDF-SHA256 with domain separation
    let hkdf = Hkdf::<Sha256>::new(None, token.as_bytes());
    let mut key = [0u8; 32];
    hkdf.expand(domain, &mut key)
        .map_err(|e| SecretError::KeyDerivation(e.to_string()))?;
    
    tracing::info!(
        domain = %String::from_utf8_lossy(domain),
        "Secret key derived from token"
    );
    
    Ok(SecretKey::new(key))
}

impl SecretKey {
    /// Derive a key from token (convenience method)
    ///
    /// See [`derive_key_from_token`] for details.
    ///
    /// # Errors
    ///
    /// See [`derive_key_from_token`] for error conditions.
    pub fn derive_from_token(token: &str, domain: &[u8]) -> Result<Self> {
        derive_key_from_token(token, domain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_derive_key() {
        let key = derive_key_from_token("test-token", b"test-domain").unwrap();
        assert_eq!(key.as_bytes().len(), 32);
    }
    
    #[test]
    fn test_derive_deterministic() {
        let key1 = derive_key_from_token("test-token", b"test-domain").unwrap();
        let key2 = derive_key_from_token("test-token", b"test-domain").unwrap();
        assert_eq!(key1.as_bytes(), key2.as_bytes());
    }
    
    #[test]
    fn test_derive_different_domains() {
        let key1 = derive_key_from_token("test-token", b"domain-v1").unwrap();
        let key2 = derive_key_from_token("test-token", b"domain-v2").unwrap();
        assert_ne!(key1.as_bytes(), key2.as_bytes());
    }
    
    #[test]
    fn test_reject_empty_token() {
        let result = derive_key_from_token("", b"domain");
        assert!(result.is_err());
    }
    
    #[test]
    fn test_reject_empty_domain() {
        let result = derive_key_from_token("token", b"");
        assert!(result.is_err());
    }
}
