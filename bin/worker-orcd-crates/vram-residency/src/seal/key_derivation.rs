//! Seal key derivation
//!
//! Derives seal keys from worker API tokens using HKDF-SHA256.

use crate::error::{Result, VramError};
use hkdf::Hkdf;
use sha2::Sha256;

/// Derive seal key from worker API token
///
/// # Arguments
///
/// * `worker_token` - Worker API token (input key material)
/// * `domain` - Domain separation string (e.g., b"llorch-seal-key-v1")
///
/// # Returns
///
/// 32-byte seal key
///
/// # Security
///
/// - Uses HKDF-SHA256 for key derivation (RFC 5869)
/// - Domain separation prevents key reuse across contexts
/// - Constant-time operations
///
/// # Errors
///
/// Returns error if worker_token is empty or HKDF expansion fails
///
/// # Example
///
/// ```no_run
/// use vram_residency::seal::derive_seal_key;
///
/// let worker_token = "worker-secret-token-here";
/// let seal_key = derive_seal_key(worker_token, b"llorch-seal-key-v1")?;
/// # Ok::<(), vram_residency::VramError>(())
/// ```
pub fn derive_seal_key(worker_token: &str, domain: &[u8]) -> Result<Vec<u8>> {
    // Validate inputs
    if worker_token.is_empty() {
        return Err(VramError::ConfigError(
            "worker_token cannot be empty".to_string(),
        ));
    }
    
    if domain.is_empty() {
        return Err(VramError::ConfigError(
            "domain cannot be empty".to_string(),
        ));
    }
    
    // HKDF-SHA256 key derivation
    // - IKM (Input Key Material): worker_token
    // - Salt: domain (for domain separation)
    // - Info: empty (no additional context needed)
    let hkdf = Hkdf::<Sha256>::new(Some(domain), worker_token.as_bytes());
    
    // Expand to 32 bytes (256 bits)
    let mut seal_key = vec![0u8; 32];
    hkdf.expand(&[], &mut seal_key)
        .map_err(|e| VramError::ConfigError(format!("HKDF expansion failed: {}", e)))?;
    
    tracing::debug!(
        domain = %String::from_utf8_lossy(domain),
        "Derived seal key via HKDF-SHA256"
    );
    
    Ok(seal_key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derive_seal_key_valid_inputs() {
        let result = derive_seal_key("test-worker-token", b"test-domain");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 32);
    }

    #[test]
    fn test_derive_seal_key_empty_token_rejected() {
        let result = derive_seal_key("", b"test-domain");
        assert!(result.is_err());
        assert!(matches!(result, Err(VramError::ConfigError(_))));
    }

    #[test]
    fn test_derive_seal_key_empty_domain_rejected() {
        let result = derive_seal_key("test-token", b"");
        assert!(result.is_err());
        assert!(matches!(result, Err(VramError::ConfigError(_))));
    }

    #[test]
    fn test_derive_seal_key_deterministic() {
        let key1 = derive_seal_key("test-token", b"domain").unwrap();
        let key2 = derive_seal_key("test-token", b"domain").unwrap();
        assert_eq!(key1, key2);
    }

    #[test]
    fn test_derive_seal_key_different_tokens_different_keys() {
        let key1 = derive_seal_key("token1", b"domain").unwrap();
        let key2 = derive_seal_key("token2", b"domain").unwrap();
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_derive_seal_key_different_domains_different_keys() {
        let key1 = derive_seal_key("test-token", b"domain1").unwrap();
        let key2 = derive_seal_key("test-token", b"domain2").unwrap();
        assert_ne!(key1, key2);
    }

    #[test]
    fn test_derive_seal_key_output_length() {
        let key = derive_seal_key("test-token", b"domain").unwrap();
        assert_eq!(key.len(), 32); // 256 bits
    }

    #[test]
    fn test_derive_seal_key_long_token() {
        let long_token = "a".repeat(1000);
        let result = derive_seal_key(&long_token, b"domain");
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 32);
    }

    #[test]
    fn test_derive_seal_key_long_domain() {
        let long_domain = vec![0x42u8; 1000];
        let result = derive_seal_key("test-token", &long_domain);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 32);
    }
}
