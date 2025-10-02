//! HMAC-SHA256 seal signature computation
//!
//! Computes and verifies cryptographic signatures for sealed shards.

use crate::error::{Result, VramError};
use crate::types::SealedShard;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use subtle::ConstantTimeEq;

type HmacSha256 = Hmac<Sha256>;

/// Compute HMAC-SHA256 signature for a sealed shard
///
/// # Arguments
///
/// * `shard` - The sealed shard to sign
/// * `seal_key` - The seal key (32 bytes)
///
/// # Returns
///
/// HMAC-SHA256 signature (32 bytes)
///
/// # Security
///
/// - Uses HMAC-SHA256 (FIPS 140-2 approved)
/// - Covers: shard_id, digest, sealed_at, gpu_device
/// - Timing-safe verification
///
/// # Errors
///
/// Returns error if seal_key is invalid or HMAC computation fails
pub fn compute_signature(shard: &SealedShard, seal_key: &[u8]) -> Result<Vec<u8>> {
    // Validate seal key length
    if seal_key.len() < 32 {
        return Err(VramError::ConfigError(
            "seal key must be at least 32 bytes".to_string(),
        ));
    }
    
    // Create HMAC instance
    let mut mac = HmacSha256::new_from_slice(seal_key)
        .map_err(|e| VramError::ConfigError(format!("HMAC initialization failed: {}", e)))?;
    
    // Sign: shard_id || digest || sealed_at || gpu_device
    mac.update(shard.shard_id.as_bytes());
    mac.update(shard.digest.as_bytes());
    
    // Convert SystemTime to u64 (seconds since UNIX_EPOCH)
    let timestamp = shard
        .sealed_at
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|_e| VramError::IntegrityViolation)?
        .as_secs();
    mac.update(&timestamp.to_le_bytes());
    
    mac.update(&shard.gpu_device.to_le_bytes());
    mac.update(&shard.vram_bytes.to_le_bytes());
    
    // Finalize and return signature
    let result = mac.finalize();
    Ok(result.into_bytes().to_vec())
}

/// Verify HMAC-SHA256 signature for a sealed shard
///
/// # Arguments
///
/// * `shard` - The sealed shard to verify
/// * `signature` - The signature to verify
/// * `seal_key` - The seal key (32 bytes)
///
/// # Returns
///
/// `Ok(())` if signature is valid, error otherwise
///
/// # Security
///
/// - Uses timing-safe comparison (via subtle crate)
/// - Re-computes signature and compares
///
/// # Errors
///
/// Returns `SealVerificationFailed` if signature doesn't match
pub fn verify_signature(
    shard: &SealedShard,
    signature: &[u8],
    seal_key: &[u8],
) -> Result<()> {
    // Re-compute signature
    let expected = compute_signature(shard, seal_key)?;
    
    // Timing-safe comparison
    if expected.len() != signature.len() {
        tracing::error!(
            shard_id = %shard.shard_id,
            expected_len = %expected.len(),
            actual_len = %signature.len(),
            "Seal signature length mismatch"
        );
        return Err(VramError::SealVerificationFailed);
    }
    
    let is_valid = expected.ct_eq(signature);
    
    if is_valid.into() {
        tracing::debug!(
            shard_id = %shard.shard_id,
            "Seal signature verified"
        );
        Ok(())
    } else {
        tracing::error!(
            shard_id = %shard.shard_id,
            "Seal signature verification failed"
        );
        Err(VramError::SealVerificationFailed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SealedShard;

    fn create_test_shard() -> SealedShard {
        SealedShard::new(
            "test-shard".to_string(),
            0,
            1024,
            "a".repeat(64),
            0x1000,
        )
    }

    #[test]
    fn test_compute_signature_valid_inputs() {
        let shard = create_test_shard();
        let seal_key = vec![0x42u8; 32];
        
        let result = compute_signature(&shard, &seal_key);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 32); // HMAC-SHA256 = 32 bytes
    }

    #[test]
    fn test_compute_signature_short_key_rejected() {
        let shard = create_test_shard();
        let short_key = vec![0x42u8; 16]; // Only 16 bytes
        
        let result = compute_signature(&shard, &short_key);
        assert!(result.is_err());
        assert!(matches!(result, Err(VramError::ConfigError(_))));
    }

    #[test]
    fn test_compute_signature_deterministic() {
        let shard = create_test_shard();
        let seal_key = vec![0x42u8; 32];
        
        let sig1 = compute_signature(&shard, &seal_key).unwrap();
        let sig2 = compute_signature(&shard, &seal_key).unwrap();
        
        assert_eq!(sig1, sig2);
    }

    #[test]
    fn test_verify_signature_valid() {
        let shard = create_test_shard();
        let seal_key = vec![0x42u8; 32];
        
        let signature = compute_signature(&shard, &seal_key).unwrap();
        let result = verify_signature(&shard, &signature, &seal_key);
        
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_signature_invalid_rejected() {
        let shard = create_test_shard();
        let seal_key = vec![0x42u8; 32];
        let invalid_sig = vec![0x00u8; 32];
        
        let result = verify_signature(&shard, &invalid_sig, &seal_key);
        assert!(result.is_err());
        assert!(matches!(result, Err(VramError::SealVerificationFailed)));
    }

    #[test]
    fn test_verify_signature_wrong_key_rejected() {
        let shard = create_test_shard();
        let seal_key1 = vec![0x42u8; 32];
        let seal_key2 = vec![0x43u8; 32];
        
        let signature = compute_signature(&shard, &seal_key1).unwrap();
        let result = verify_signature(&shard, &signature, &seal_key2);
        
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_signature_tampered_shard_id_rejected() {
        let mut shard = create_test_shard();
        let seal_key = vec![0x42u8; 32];
        
        let signature = compute_signature(&shard, &seal_key).unwrap();
        
        // Tamper with shard_id
        shard.shard_id = "tampered".to_string();
        
        let result = verify_signature(&shard, &signature, &seal_key);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_signature_tampered_digest_rejected() {
        let mut shard = create_test_shard();
        let seal_key = vec![0x42u8; 32];
        
        let signature = compute_signature(&shard, &seal_key).unwrap();
        
        // Tamper with digest
        shard.digest = "b".repeat(64);
        
        let result = verify_signature(&shard, &signature, &seal_key);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_signature_length_mismatch_rejected() {
        let shard = create_test_shard();
        let seal_key = vec![0x42u8; 32];
        let wrong_length_sig = vec![0x00u8; 16]; // Wrong length
        
        let result = verify_signature(&shard, &wrong_length_sig, &seal_key);
        assert!(result.is_err());
    }

    #[test]
    fn test_signature_different_for_different_shards() {
        let shard1 = create_test_shard();
        let mut shard2 = create_test_shard();
        shard2.shard_id = "different-shard".to_string();
        
        let seal_key = vec![0x42u8; 32];
        
        let sig1 = compute_signature(&shard1, &seal_key).unwrap();
        let sig2 = compute_signature(&shard2, &seal_key).unwrap();
        
        assert_ne!(sig1, sig2);
    }

    #[test]
    fn test_signature_different_for_different_keys() {
        let shard = create_test_shard();
        let key1 = vec![0x42u8; 32];
        let key2 = vec![0x43u8; 32];
        
        let sig1 = compute_signature(&shard, &key1).unwrap();
        let sig2 = compute_signature(&shard, &key2).unwrap();
        
        assert_ne!(sig1, sig2);
    }

    #[test]
    fn test_timing_safe_comparison() {
        // This test verifies that verification uses constant-time comparison
        // We can't easily test timing, but we can verify the behavior is consistent
        let shard = create_test_shard();
        let seal_key = vec![0x42u8; 32];
        
        let valid_sig = compute_signature(&shard, &seal_key).unwrap();
        let mut invalid_sig = valid_sig.clone();
        invalid_sig[0] ^= 0x01; // Flip one bit
        
        // Both should fail/succeed consistently
        assert!(verify_signature(&shard, &valid_sig, &seal_key).is_ok());
        assert!(verify_signature(&shard, &invalid_sig, &seal_key).is_err());
    }
}
