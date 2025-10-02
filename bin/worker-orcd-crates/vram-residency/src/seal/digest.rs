//! SHA-256 digest computation
//!
//! Computes SHA-256 digests of model data.

use sha2::{Digest, Sha256};
use crate::cuda_ffi::SafeCudaPtr;
use crate::error::Result;

/// Compute SHA-256 digest of data
///
/// # Arguments
///
/// * `data` - The data to hash
///
/// # Returns
///
/// SHA-256 digest as hex string (64 characters)
pub fn compute_digest(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    format!("{:x}", hasher.finalize())
}

/// Re-compute digest from VRAM contents
///
/// # Arguments
///
/// * `cuda_ptr` - Safe CUDA pointer to VRAM data
///
/// # Returns
///
/// SHA-256 digest as hex string
///
/// # Errors
///
/// Returns error if VRAM read fails
///
/// # Security
///
/// This function reads from VRAM via CUDA FFI to verify integrity.
/// Uses bounds-checked SafeCudaPtr to prevent out-of-bounds access.
pub fn recompute_digest_from_vram(cuda_ptr: &SafeCudaPtr) -> Result<String> {
    // Read entire VRAM contents
    let data = cuda_ptr.read_at(0, cuda_ptr.size())?;
    
    // Compute SHA-256 digest
    let mut hasher = Sha256::new();
    hasher.update(&data);
    let digest = format!("{:x}", hasher.finalize());
    
    tracing::debug!(
        size = %cuda_ptr.size(),
        digest = %&digest[..16],
        "Re-computed digest from VRAM"
    );
    
    Ok(digest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_digest_empty_data() {
        let digest = compute_digest(&[]);
        assert_eq!(digest.len(), 64); // SHA-256 = 64 hex chars
        // SHA-256 of empty string
        assert_eq!(digest, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    }

    #[test]
    fn test_compute_digest_small_data() {
        let digest = compute_digest(b"hello");
        assert_eq!(digest.len(), 64);
        // SHA-256 of "hello"
        assert_eq!(digest, "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824");
    }

    #[test]
    fn test_compute_digest_deterministic() {
        let data = b"test data for determinism";
        let digest1 = compute_digest(data);
        let digest2 = compute_digest(data);
        assert_eq!(digest1, digest2);
    }

    #[test]
    fn test_compute_digest_different_data_different_output() {
        let digest1 = compute_digest(b"data1");
        let digest2 = compute_digest(b"data2");
        assert_ne!(digest1, digest2);
    }

    #[test]
    fn test_compute_digest_large_data() {
        let data = vec![0x42u8; 1024 * 1024]; // 1MB
        let digest = compute_digest(&data);
        assert_eq!(digest.len(), 64);
    }

    #[test]
    fn test_compute_digest_format_is_hex() {
        let digest = compute_digest(b"test");
        assert!(digest.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(digest.chars().all(|c| c.is_lowercase() || c.is_ascii_digit()));
    }

    #[test]
    fn test_compute_digest_collision_resistance() {
        // Different inputs should produce different outputs
        let digest1 = compute_digest(b"abc");
        let digest2 = compute_digest(b"abd");
        assert_ne!(digest1, digest2);
    }

    #[test]
    fn test_compute_digest_avalanche_effect() {
        // Single bit change should drastically change output
        let digest1 = compute_digest(&[0x00]);
        let digest2 = compute_digest(&[0x01]);
        assert_ne!(digest1, digest2);
        
        // Count different characters
        let diff_count = digest1.chars()
            .zip(digest2.chars())
            .filter(|(a, b)| a != b)
            .count();
        
        // Should have significant differences (avalanche effect)
        assert!(diff_count > 30);
    }
}
