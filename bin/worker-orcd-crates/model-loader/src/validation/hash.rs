//! SHA-256 hash verification
//!
//! Implements HASH-001 to HASH-007 from 20_security.md

use crate::error::{LoadError, Result};
use sha2::{Digest, Sha256};

/// Compute SHA-256 hash of bytes
///
/// Returns lowercase hex string (64 characters)
pub fn compute_hash(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

/// Verify hash matches expected value
///
/// # Security Requirements
/// - HASH-001: SHA-256 MUST be used (FIPS 140-2 approved)
/// - HASH-002: Hash format MUST be validated (64 hex chars)
/// - HASH-003: Hash computation MUST occur before GGUF parsing
/// - HASH-004: Hash mismatch MUST reject model load
/// - HASH-006: Computed hash MUST be logged for audit trail
/// - HASH-007: Hash validation MUST use input-validation crate
pub fn verify_hash(bytes: &[u8], expected_hash: &str) -> Result<()> {
    // Validate hash format (HASH-002) using input-validation
    input_validation::validate_hex_string(expected_hash, 64)
        .map_err(|e| LoadError::InvalidFormat(
            format!("Invalid hash format: {}", e)
        ))?;
    
    // Compute actual hash (HASH-001: SHA-256)
    let actual_hash = compute_hash(bytes);
    
    // Compare (HASH-004)
    if actual_hash != expected_hash {
        return Err(LoadError::HashMismatch {
            expected: expected_hash.to_string(),
            actual: actual_hash,
        });
    }
    
    // Log for audit trail (HASH-006)
    tracing::info!(
        hash = %actual_hash,
        "Model hash verified"
    );
    
    Ok(())
}

// TODO(Post-M0): Add streaming hash computation per 30_dependencies.md §9.1
// #[cfg(feature = "streaming")]
// pub async fn compute_hash_streaming(reader: impl AsyncRead) -> Result<String> {
//     let mut hasher = Sha256::new();
//     let mut buffer = [0u8; 8192];
//     
//     loop {
//         let n = reader.read(&mut buffer).await?;
//         if n == 0 {
//             break;
//         }
//         hasher.update(&buffer[..n]);
//     }
//     
//     Ok(format!("{:x}", hasher.finalize()))
// }

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_compute_hash() {
        let bytes = b"test data";
        let hash = compute_hash(bytes);
        
        // SHA-256 of "test data"
        assert_eq!(hash.len(), 64);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }
    
    #[test]
    fn test_verify_hash_success() {
        let bytes = b"test data";
        let hash = compute_hash(bytes);
        
        assert!(verify_hash(bytes, &hash).is_ok());
    }
    
    #[test]
    fn test_verify_hash_mismatch() {
        let bytes = b"test data";
        let wrong_hash = "0".repeat(64);
        
        let result = verify_hash(bytes, &wrong_hash);
        assert!(matches!(result, Err(LoadError::HashMismatch { .. })));
    }
}
