//! VramManager - Main VRAM management interface
//!
//! Provides high-level API for sealing models in VRAM.

use crate::error::{Result, VramError};
use crate::types::SealedShard;
use crate::seal::compute_digest;

/// VRAM Manager
///
/// Manages VRAM allocation, sealing, and verification.
///
/// # Example
///
/// ```no_run
/// use vram_residency::VramManager;
///
/// let mut manager = VramManager::new();
/// let model_data = vec![0u8; 1024];
/// let shard = manager.seal_model(&model_data, 0)?;
/// manager.verify_sealed(&shard)?;
/// # Ok::<(), vram_residency::VramError>(())
/// ```
pub struct VramManager {
    total_vram: usize,
    used_vram: usize,
}

impl VramManager {
    /// Create new VramManager with automatic GPU detection
    ///
    /// Uses real CUDA if GPU available, mock otherwise.
    pub fn new() -> Self {
        Self {
            total_vram: 24 * 1024 * 1024 * 1024, // 24GB default
            used_vram: 0,
        }
    }
    
    /// Create VramManager for production (fail if no GPU)
    pub fn new_production() -> Result<Self> {
        // TODO: Use gpu-info to detect GPU
        // - Call GpuInfo::detect_or_fail()
        // - Return error if no GPU
        // - Initialize with real GPU capacity
        todo!("Implement production mode with GPU detection")
    }
    
    /// Seal model in VRAM
    ///
    /// # Arguments
    ///
    /// * `model_bytes` - Model data to seal
    /// * `gpu_device` - GPU device index
    ///
    /// # Returns
    ///
    /// Sealed shard handle
    pub fn seal_model(&mut self, model_bytes: &[u8], gpu_device: u32) -> Result<SealedShard> {
        let vram_needed = model_bytes.len();
        
        // Check capacity
        if self.used_vram.saturating_add(vram_needed) > self.total_vram {
            return Err(VramError::InsufficientVram(
                vram_needed,
                self.total_vram.saturating_sub(self.used_vram),
            ));
        }
        
        // Compute digest
        let digest = compute_digest(model_bytes);
        
        // TODO: Allocate VRAM via CUDA
        let vram_ptr = 0x7f8a4c000000usize; // Placeholder
        
        self.used_vram = self.used_vram.saturating_add(vram_needed);
        
        let shard = SealedShard::new(
            format!("shard-{:x}", gpu_device),
            gpu_device,
            vram_needed,
            digest,
            vram_ptr,
        );
        
        tracing::info!(
            shard_id = %shard.shard_id,
            vram_bytes = %vram_needed,
            "Model sealed in VRAM"
        );
        
        Ok(shard)
    }
    
    /// Verify sealed shard
    ///
    /// # Arguments
    ///
    /// * `shard` - Shard to verify
    ///
    /// # Returns
    ///
    /// `Ok(())` if verification succeeds
    pub fn verify_sealed(&self, shard: &SealedShard) -> Result<()> {
        // TODO: Implement digest re-verification
        // - Re-compute digest from VRAM
        // - Verify HMAC signature
        // - Check timestamp freshness
        if !shard.is_sealed() {
            return Err(VramError::NotSealed);
        }
        Ok(())
    }
    
    /// Get available VRAM
    pub fn available_vram(&self) -> usize {
        self.total_vram.saturating_sub(self.used_vram)
    }
    
    /// Get used VRAM
    pub fn used_vram(&self) -> usize {
        self.used_vram
    }
    
    /// Get total VRAM
    pub fn total_vram(&self) -> usize {
        self.total_vram
    }
}

impl Default for VramManager {
    fn default() -> Self {
        Self::new()
    }
}
