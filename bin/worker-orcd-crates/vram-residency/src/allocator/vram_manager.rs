//! VramManager - Main VRAM management interface
//!
//! Provides high-level API for sealing models in VRAM with cryptographic integrity.

use crate::error::{Result, VramError};
use crate::types::SealedShard;
use crate::seal::{compute_digest, compute_signature, verify_signature, derive_seal_key};
use crate::cuda_ffi::{CudaContext, SafeCudaPtr};
use crate::audit::events::{emit_vram_sealed, emit_seal_verified, emit_seal_verification_failed};
use crate::validation::{validate_shard_id, validate_gpu_device};
use std::collections::HashMap;

/// VRAM Manager
///
/// Manages VRAM allocation, sealing, and cryptographic verification.
///
/// # Security
///
/// - All models sealed with HMAC-SHA256 signatures
/// - VRAM integrity verified before each execution
/// - Seal keys derived via HKDF-SHA256 from worker token
///
/// # Example
///
/// ```no_run
/// use vram_residency::VramManager;
///
/// let mut manager = VramManager::new_with_token("worker-token", 0)?;
/// let model_data = vec![0u8; 1024];
/// let shard = manager.seal_model(&model_data, 0)?;
/// manager.verify_sealed(&shard)?;
/// # Ok::<(), vram_residency::VramError>(())
/// ```
pub struct VramManager {
    context: CudaContext,
    seal_key: Vec<u8>,
    allocations: HashMap<usize, SafeCudaPtr>,
}

impl VramManager {
    /// Create new VramManager with mock CUDA (for testing)
    ///
    /// # Warning
    ///
    /// This is for testing only. Use `new_with_token()` for production.
    pub fn new() -> Self {
        let context = CudaContext::new(0)
            .expect("Failed to create CUDA context in test mode");
        
        // Mock seal key for testing (DO NOT use in production)
        let seal_key = vec![0x42u8; 32];
        
        Self {
            context,
            seal_key,
            allocations: HashMap::new(),
        }
    }
    
    /// Create VramManager for production with worker token
    ///
    /// # Arguments
    ///
    /// * `worker_token` - Worker API token for seal key derivation
    /// * `gpu_device` - GPU device index
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - No GPU detected
    /// - Worker token is invalid
    /// - CUDA initialization fails
    ///
    /// # Security
    ///
    /// Seal key is derived via HKDF-SHA256 from worker_token with domain separation.
    pub fn new_with_token(worker_token: &str, gpu_device: u32) -> Result<Self> {
        // Initialize CUDA context (fails if no GPU in production mode)
        let context = CudaContext::new(gpu_device)?;
        
        // Derive seal key from worker token
        let seal_key = derive_seal_key(worker_token, b"llorch-vram-seal-v1")?;
        
        tracing::info!(
            gpu_device = %gpu_device,
            "VramManager initialized with CUDA context"
        );
        
        Ok(Self {
            context,
            seal_key,
            allocations: HashMap::new(),
        })
    }
    
    /// Seal model in VRAM with cryptographic signature
    ///
    /// # Arguments
    ///
    /// * `model_bytes` - Model data to seal
    /// * `gpu_device` - GPU device index
    ///
    /// # Returns
    ///
    /// Sealed shard with HMAC-SHA256 signature
    ///
    /// # Security
    ///
    /// 1. Validates inputs (shard ID, GPU device, model size)
    /// 2. Allocates VRAM via CUDA
    /// 3. Copies model data to VRAM
    /// 4. Computes SHA-256 digest
    /// 5. Generates HMAC-SHA256 seal signature
    /// 6. Returns tamper-evident sealed shard
    pub fn seal_model(&mut self, model_bytes: &[u8], gpu_device: u32) -> Result<SealedShard> {
        let vram_needed = model_bytes.len();
        
        // Validate model size (prevent zero-size and DoS)
        if vram_needed == 0 {
            return Err(VramError::InvalidInput(
                "model size cannot be zero".to_string()
            ));
        }
        
        // Validate GPU device index
        // Note: In production mode, CudaContext already validates this
        // This is defense-in-depth
        #[cfg(not(test))]
        {
            let gpu_info = gpu_info::detect_gpus();
            validate_gpu_device(gpu_device, gpu_info.count as u32)?;
        }
        
        // Check capacity
        let available = self.context.get_free_vram()?;
        if vram_needed > available {
            return Err(VramError::InsufficientVram(vram_needed, available));
        }
        
        // Allocate VRAM via CUDA
        let mut cuda_ptr = self.context.allocate_vram(vram_needed)?;
        
        // Copy model data to VRAM
        cuda_ptr.write_at(0, model_bytes)?;
        
        // Compute SHA-256 digest
        let digest = compute_digest(model_bytes);
        
        // Create sealed shard
        let vram_ptr = cuda_ptr.as_ptr() as usize;
        let shard_id = format!("shard-{:x}-{:x}", gpu_device, vram_ptr);
        
        // Validate generated shard ID (defense-in-depth)
        validate_shard_id(&shard_id)?;
        
        let mut shard = SealedShard::new(
            shard_id,
            gpu_device,
            vram_needed,
            digest,
            vram_ptr,
        );
        
        // Compute HMAC-SHA256 signature
        let signature = compute_signature(&shard, &self.seal_key)?;
        shard.set_signature(signature);
        
        // Store allocation
        self.allocations.insert(vram_ptr, cuda_ptr);
        
        // Emit audit event
        emit_vram_sealed(&shard);
        
        tracing::info!(
            shard_id = %shard.shard_id,
            vram_bytes = %vram_needed,
            gpu_device = %gpu_device,
            "Model sealed in VRAM with cryptographic signature"
        );
        
        Ok(shard)
    }
    
    /// Verify sealed shard integrity
    ///
    /// # Arguments
    ///
    /// * `shard` - Shard to verify
    ///
    /// # Returns
    ///
    /// `Ok(())` if verification succeeds
    ///
    /// # Security
    ///
    /// 1. Verifies HMAC-SHA256 signature (timing-safe)
    /// 2. Re-computes digest from VRAM contents
    /// 3. Compares with original digest
    ///
    /// # Errors
    ///
    /// Returns `SealVerificationFailed` if:
    /// - Signature is invalid
    /// - Digest doesn't match (VRAM corruption)
    /// - Shard not properly sealed
    pub fn verify_sealed(&self, shard: &SealedShard) -> Result<()> {
        // Check if shard is sealed
        if !shard.is_sealed() {
            return Err(VramError::NotSealed);
        }
        
        // Verify HMAC signature
        let signature = shard.signature()
            .ok_or(VramError::NotSealed)?;
        verify_signature(shard, signature, &self.seal_key)?;
        
        // Get VRAM allocation
        let cuda_ptr = self.allocations.get(&shard.vram_ptr())
            .ok_or(VramError::IntegrityViolation)?;
        
        // Re-compute digest from VRAM
        let vram_digest = crate::seal::digest::recompute_digest_from_vram(cuda_ptr)?;
        
        // Compare digests
        if vram_digest != shard.digest {
            let reason = format!(
                "digest mismatch: expected {}, got {}",
                &shard.digest[..16],
                &vram_digest[..16]
            );
            
            // Emit critical audit event
            emit_seal_verification_failed(shard, &reason);
            
            tracing::error!(
                shard_id = %shard.shard_id,
                expected = %shard.digest,
                actual = %vram_digest,
                "VRAM digest mismatch - corruption detected"
            );
            return Err(VramError::SealVerificationFailed);
        }
        
        // Emit audit event for successful verification
        emit_seal_verified(shard);
        
        tracing::debug!(
            shard_id = %shard.shard_id,
            "Seal verification passed"
        );
        
        Ok(())
    }
    
    /// Get available VRAM
    pub fn available_vram(&self) -> Result<usize> {
        self.context.get_free_vram()
    }
    
    /// Get total VRAM
    pub fn total_vram(&self) -> Result<usize> {
        self.context.get_total_vram()
    }
}

impl Default for VramManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::SystemTime;

    #[test]
    fn test_new_manager_creation() {
        let manager = VramManager::new();
        assert!(manager.available_vram().is_ok());
        assert!(manager.total_vram().is_ok());
    }

    #[test]
    fn test_seal_model_basic() {
        let mut manager = VramManager::new();
        let model_data = vec![0x42u8; 1024];
        
        let result = manager.seal_model(&model_data, 0);
        assert!(result.is_ok());
        
        let shard = result.unwrap();
        assert_eq!(shard.vram_bytes, 1024);
        assert_eq!(shard.digest.len(), 64);
        assert!(shard.is_sealed());
    }

    #[test]
    fn test_seal_model_zero_size_rejected() {
        let mut manager = VramManager::new();
        let empty_data = vec![];
        
        let result = manager.seal_model(&empty_data, 0);
        assert!(result.is_err());
        assert!(matches!(result, Err(VramError::InvalidInput(_))));
    }

    #[test]
    fn test_verify_sealed_valid() {
        let mut manager = VramManager::new();
        let model_data = vec![0x42u8; 1024];
        
        let shard = manager.seal_model(&model_data, 0).unwrap();
        let result = manager.verify_sealed(&shard);
        
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_sealed_unsealed_shard_rejected() {
        let manager = VramManager::new();
        let shard = SealedShard::new(
            "test".to_string(),
            0,
            1024,
            String::new(), // Empty digest = unsealed
            0x1000,
        );
        
        let result = manager.verify_sealed(&shard);
        assert!(result.is_err());
    }

    #[test]
    fn test_verify_sealed_missing_signature_rejected() {
        let manager = VramManager::new();
        let shard = SealedShard::new(
            "test".to_string(),
            0,
            1024,
            "a".repeat(64),
            0x1000,
        );
        // No signature set
        
        let result = manager.verify_sealed(&shard);
        assert!(result.is_err());
        assert!(matches!(result, Err(VramError::NotSealed)));
    }

    #[test]
    fn test_multiple_allocations() {
        let mut manager = VramManager::new();
        
        let shard1 = manager.seal_model(&vec![0x11u8; 1024], 0).unwrap();
        let shard2 = manager.seal_model(&vec![0x22u8; 2048], 0).unwrap();
        let shard3 = manager.seal_model(&vec![0x33u8; 4096], 0).unwrap();
        
        assert_eq!(shard1.vram_bytes, 1024);
        assert_eq!(shard2.vram_bytes, 2048);
        assert_eq!(shard3.vram_bytes, 4096);
        
        // All should verify
        assert!(manager.verify_sealed(&shard1).is_ok());
        assert!(manager.verify_sealed(&shard2).is_ok());
        assert!(manager.verify_sealed(&shard3).is_ok());
    }

    #[test]
    fn test_seal_generates_unique_shard_ids() {
        let mut manager = VramManager::new();
        
        let shard1 = manager.seal_model(&vec![0x11u8; 1024], 0).unwrap();
        let shard2 = manager.seal_model(&vec![0x22u8; 1024], 0).unwrap();
        
        assert_ne!(shard1.shard_id, shard2.shard_id);
    }

    #[test]
    fn test_seal_computes_correct_digest() {
        let mut manager = VramManager::new();
        let data = b"test data";
        
        let shard = manager.seal_model(data, 0).unwrap();
        
        // Verify digest matches expected SHA-256
        use crate::seal::compute_digest;
        let expected_digest = compute_digest(data);
        assert_eq!(shard.digest, expected_digest);
    }

    #[test]
    fn test_seal_sets_signature() {
        let mut manager = VramManager::new();
        let data = vec![0x42u8; 1024];
        
        let shard = manager.seal_model(&data, 0).unwrap();
        
        assert!(shard.signature().is_some());
        assert_eq!(shard.signature().unwrap().len(), 32); // HMAC-SHA256
    }

    #[test]
    fn test_available_vram_returns_value() {
        let manager = VramManager::new();
        let available = manager.available_vram();
        
        assert!(available.is_ok());
        assert!(available.unwrap() > 0);
    }

    #[test]
    fn test_total_vram_returns_value() {
        let manager = VramManager::new();
        let total = manager.total_vram();
        
        assert!(total.is_ok());
        assert!(total.unwrap() > 0);
    }

    #[test]
    fn test_total_vram_greater_than_or_equal_available() {
        let manager = VramManager::new();
        let total = manager.total_vram().unwrap();
        let available = manager.available_vram().unwrap();
        
        assert!(total >= available);
    }

    #[test]
    fn test_seal_different_data_different_digests() {
        let mut manager = VramManager::new();
        
        let shard1 = manager.seal_model(&vec![0x11u8; 1024], 0).unwrap();
        let shard2 = manager.seal_model(&vec![0x22u8; 1024], 0).unwrap();
        
        assert_ne!(shard1.digest, shard2.digest);
    }

    #[test]
    fn test_seal_same_data_same_digest() {
        let mut manager = VramManager::new();
        let data = vec![0x42u8; 1024];
        
        let shard1 = manager.seal_model(&data, 0).unwrap();
        let shard2 = manager.seal_model(&data, 0).unwrap();
        
        assert_eq!(shard1.digest, shard2.digest);
    }

    #[test]
    fn test_seal_sets_gpu_device() {
        let mut manager = VramManager::new();
        let data = vec![0x42u8; 1024];
        
        let shard = manager.seal_model(&data, 0).unwrap();
        assert_eq!(shard.gpu_device, 0);
    }

    #[test]
    fn test_seal_sets_vram_bytes() {
        let mut manager = VramManager::new();
        let data = vec![0x42u8; 2048];
        
        let shard = manager.seal_model(&data, 0).unwrap();
        assert_eq!(shard.vram_bytes, 2048);
    }

    #[test]
    fn test_seal_sets_timestamp() {
        let mut manager = VramManager::new();
        let data = vec![0x42u8; 1024];
        
        let before = SystemTime::now();
        let shard = manager.seal_model(&data, 0).unwrap();
        let after = SystemTime::now();
        
        assert!(shard.sealed_at >= before);
        assert!(shard.sealed_at <= after);
    }

    #[test]
    fn test_default_trait() {
        let manager = VramManager::default();
        assert!(manager.available_vram().is_ok());
    }

    #[test]
    fn test_large_model_seal() {
        let mut manager = VramManager::new();
        let large_data = vec![0x42u8; 10 * 1024 * 1024]; // 10MB
        
        let result = manager.seal_model(&large_data, 0);
        // May succeed or fail depending on available VRAM
        // Just verify it doesn't panic
        let _ = result;
    }

    #[test]
    fn test_seal_validates_shard_id_format() {
        let mut manager = VramManager::new();
        let data = vec![0x42u8; 1024];
        
        let shard = manager.seal_model(&data, 0).unwrap();
        
        // Shard ID should be valid (no path traversal, etc.)
        use crate::validation::validate_shard_id;
        assert!(validate_shard_id(&shard.shard_id).is_ok());
    }
}
