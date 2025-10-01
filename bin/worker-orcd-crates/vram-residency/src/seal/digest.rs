//! SHA-256 digest computation
//!
//! Computes SHA-256 digests of model data.

use sha2::{Digest, Sha256};

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
/// * `vram_ptr` - VRAM pointer
/// * `size` - Size of data
///
/// # Returns
///
/// SHA-256 digest as hex string
///
/// # Safety
///
/// This function will read from VRAM via CUDA FFI.
/// Caller must ensure pointer is valid.
pub fn recompute_digest_from_vram(_vram_ptr: usize, _size: usize) -> String {
    // TODO: Implement VRAM digest re-computation
    // - Read data from VRAM via cudaMemcpy
    // - Compute SHA-256 digest
    // - Return hex string
    todo!("Implement VRAM digest re-computation")
}
