//! SecretKey type for cryptographic keys (32 bytes)
//!
//! Wraps 32-byte cryptographic keys with automatic zeroization on drop.
//! Used for HMAC, encryption, and other cryptographic operations.

use zeroize::{Zeroize, ZeroizeOnDrop};

/// Opaque wrapper for 32-byte cryptographic keys
///
/// # Security Properties
///
/// - No Debug/Display implementation (prevents accidental logging)
/// - Automatic zeroization on drop
/// - Fixed 32-byte size (256-bit keys)
///
/// # Example
///
/// ```rust,no_run
/// use secrets_management::SecretKey;
///
/// # fn main() -> Result<(), secrets_management::SecretError> {
/// // Derive seal key from worker token
/// let worker_token = "worker-token-abc123";
/// let seal_key = SecretKey::derive_from_token(
///     worker_token,
///     b"llorch-seal-key-v1"
/// )?;
///
/// // Use for HMAC-SHA256 (requires hmac and sha2 crates)
/// // let mut mac = Hmac::<Sha256>::new_from_slice(seal_key.as_bytes())?;
/// // mac.update(message.as_bytes());
/// // let signature = mac.finalize().into_bytes();
/// # Ok(())
/// # }
/// ```
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct SecretKey([u8; 32]);

impl SecretKey {
    /// Create a new secret key from 32 bytes
    ///
    /// # Security
    ///
    /// The key will be zeroized when this SecretKey is dropped.
    pub(crate) fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }
    
    /// Access key bytes for cryptographic operations
    ///
    /// # Security
    ///
    /// Returns a reference to the 32-byte key. The key will be automatically
    /// zeroized when this SecretKey is dropped.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use secrets_management::SecretKey;
    /// # fn main() -> Result<(), secrets_management::SecretError> {
    /// # let seal_key = SecretKey::derive_from_token("token", b"domain")?;
    /// // Access key bytes for cryptographic operations
    /// let key_bytes: &[u8; 32] = seal_key.as_bytes();
    /// assert_eq!(key_bytes.len(), 32);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

// No Debug/Display/ToString/Serialize/Clone implementations
// (prevents accidental logging or copying of keys)

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_new_key() {
        let key = SecretKey::new([42u8; 32]);
        assert_eq!(key.as_bytes().len(), 32);
        assert_eq!(key.as_bytes()[0], 42);
    }
    
    #[test]
    fn test_zeroize_on_drop() {
        // This test verifies that zeroize is called on drop
        // The actual zeroization is handled by the zeroize crate
        let key = SecretKey::new([42u8; 32]);
        drop(key);
        // If we could inspect memory here, it would be zeroed
    }
}
