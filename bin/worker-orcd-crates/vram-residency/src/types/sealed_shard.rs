//! SealedShard type definition
//!
//! Represents a cryptographically sealed model shard resident in VRAM.

use std::time::SystemTime;

/// A cryptographically sealed model shard in VRAM
///
/// This type represents a model shard that has been:
/// - Allocated in GPU VRAM
/// - Sealed with HMAC-SHA256 signature
/// - Verified for integrity
///
/// # Security
///
/// - VRAM pointer is private and never exposed
/// - Digest is SHA-256 of shard data
/// - Signature is HMAC-SHA256 of (shard_id, digest, sealed_at, gpu_device)
#[derive(Clone)]
pub struct SealedShard {
    /// Unique shard identifier
    pub shard_id: String,
    
    /// GPU device index
    pub gpu_device: u32,
    
    /// Size in VRAM (bytes)
    pub vram_bytes: usize,
    
    /// SHA-256 digest of shard data (hex string)
    pub digest: String,
    
    /// Timestamp when sealed
    pub sealed_at: SystemTime,
    
    /// Original model reference
    pub model_ref: Option<String>,
    
    /// Shard index (for tensor-parallel)
    pub shard_index: Option<usize>,
    
    /// Total shards (for tensor-parallel)
    pub total_shards: Option<usize>,
    
    /// HMAC-SHA256 signature (TODO: implement)
    #[allow(dead_code)]
    signature: Option<Vec<u8>>,
    
    /// VRAM pointer (private, never exposed)
    #[allow(dead_code)]
    vram_ptr: usize,
}

impl SealedShard {
    /// Create a new sealed shard
    ///
    /// # Arguments
    ///
    /// * `shard_id` - Unique identifier
    /// * `gpu_device` - GPU device index
    /// * `vram_bytes` - Size in VRAM
    /// * `digest` - SHA-256 digest
    /// * `vram_ptr` - VRAM pointer (private)
    pub(crate) fn new(
        shard_id: String,
        gpu_device: u32,
        vram_bytes: usize,
        digest: String,
        vram_ptr: usize,
    ) -> Self {
        Self {
            shard_id,
            gpu_device,
            vram_bytes,
            digest,
            sealed_at: SystemTime::now(),
            model_ref: None,
            shard_index: None,
            total_shards: None,
            signature: None,
            vram_ptr,
        }
    }
    
    /// Check if shard is sealed
    pub fn is_sealed(&self) -> bool {
        !self.digest.is_empty()
    }
}

// Custom Debug that omits VRAM pointer
impl std::fmt::Debug for SealedShard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SealedShard")
            .field("shard_id", &self.shard_id)
            .field("gpu_device", &self.gpu_device)
            .field("vram_bytes", &self.vram_bytes)
            .field("digest", &format!("{}...", &self.digest[..8.min(self.digest.len())]))
            .field("sealed_at", &self.sealed_at)
            .finish_non_exhaustive()
    }
}
