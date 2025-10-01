//! Seal key derivation
//!
//! Derives seal keys from worker API tokens using HKDF-SHA256.

use crate::error::Result;

/// Derive seal key from worker API token
///
/// # Arguments
///
/// * `worker_token` - Worker API token
/// * `domain` - Domain separation string (e.g., "llorch-seal-key-v1")
///
/// # Returns
///
/// 32-byte seal key
///
/// # Security
///
/// - Uses HKDF-SHA256 for key derivation
/// - Domain separation prevents key reuse
/// - Key is zeroized on drop (via secrets-management)
pub fn derive_seal_key(_worker_token: &str, _domain: &[u8]) -> Result<Vec<u8>> {
    // TODO: Integrate with secrets-management crate
    // - Use SecretKey::derive_from_token()
    // - Return 32-byte key
    // - Ensure zeroization on drop
    todo!("Implement seal key derivation via secrets-management")
}
