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
    
    /// HMAC-SHA256 signature (private, verified via verify_signature)
    pub(crate) signature: Option<Vec<u8>>,
    
    /// VRAM pointer (private, never exposed)
    pub(crate) vram_ptr: usize,
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
    
    /// Set signature (internal use only)
    pub(crate) fn set_signature(&mut self, signature: Vec<u8>) {
        self.signature = Some(signature);
    }
    
    /// Get signature (internal use only)
    pub(crate) fn signature(&self) -> Option<&[u8]> {
        self.signature.as_deref()
    }
    
    /// Get VRAM pointer (internal use only)
    pub(crate) fn vram_ptr(&self) -> usize {
        self.vram_ptr
    }
    
    /// Clear signature (for testing signature verification failure)
    ///
    /// # Warning
    /// This is intended for testing only. In production, signatures should never be removed.
    #[doc(hidden)]
    pub fn clear_signature_for_test(&mut self) {
        self.signature = None;
    }
    
    /// Replace signature with invalid data (for testing signature verification failure)
    ///
    /// # Warning
    /// This is intended for testing only. In production, signatures should never be tampered with.
    #[doc(hidden)]
    pub fn replace_signature_for_test(&mut self, invalid_signature: Vec<u8>) {
        self.signature = Some(invalid_signature);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_shard_creation() {
        let shard = SealedShard::new(
            "test-shard".to_string(),
            0,
            1024,
            "a".repeat(64),
            0x1000,
        );
        
        assert_eq!(shard.shard_id, "test-shard");
        assert_eq!(shard.gpu_device, 0);
        assert_eq!(shard.vram_bytes, 1024);
        assert_eq!(shard.digest.len(), 64);
        assert_eq!(shard.vram_ptr(), 0x1000);
    }

    #[test]
    fn test_is_sealed_with_digest() {
        let shard = SealedShard::new(
            "test".to_string(),
            0,
            1024,
            "a".repeat(64),
            0x1000,
        );
        assert!(shard.is_sealed());
    }

    #[test]
    fn test_is_sealed_without_digest() {
        let shard = SealedShard::new(
            "test".to_string(),
            0,
            1024,
            String::new(),
            0x1000,
        );
        assert!(!shard.is_sealed());
    }

    #[test]
    fn test_set_and_get_signature() {
        let mut shard = SealedShard::new(
            "test".to_string(),
            0,
            1024,
            "a".repeat(64),
            0x1000,
        );
        
        assert!(shard.signature().is_none());
        
        let sig = vec![0x42u8; 32];
        shard.set_signature(sig.clone());
        
        assert!(shard.signature().is_some());
        assert_eq!(shard.signature().unwrap(), sig.as_slice());
    }

    #[test]
    fn test_vram_ptr_accessor() {
        let shard = SealedShard::new(
            "test".to_string(),
            0,
            1024,
            "a".repeat(64),
            0xDEADBEEF,
        );
        assert_eq!(shard.vram_ptr(), 0xDEADBEEF);
    }

    #[test]
    fn test_clone() {
        let mut shard = SealedShard::new(
            "test".to_string(),
            0,
            1024,
            "a".repeat(64),
            0x1000,
        );
        shard.set_signature(vec![0x42u8; 32]);
        
        let cloned = shard.clone();
        assert_eq!(cloned.shard_id, shard.shard_id);
        assert_eq!(cloned.vram_ptr(), shard.vram_ptr());
        assert_eq!(cloned.signature(), shard.signature());
    }

    #[test]
    fn test_debug_format_omits_vram_ptr() {
        let shard = SealedShard::new(
            "test".to_string(),
            0,
            1024,
            "abcd1234".repeat(8),
            0xDEADBEEF,
        );
        
        let debug_str = format!("{:?}", shard);
        assert!(!debug_str.contains("DEADBEEF"));
        assert!(!debug_str.contains("vram_ptr"));
        assert!(debug_str.contains("test"));
    }

    #[test]
    fn test_optional_fields_default_to_none() {
        let shard = SealedShard::new(
            "test".to_string(),
            0,
            1024,
            "a".repeat(64),
            0x1000,
        );
        
        assert!(shard.model_ref.is_none());
        assert!(shard.shard_index.is_none());
        assert!(shard.total_shards.is_none());
        assert!(shard.signature.is_none());
    }
}
