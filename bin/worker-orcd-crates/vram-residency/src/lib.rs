//! # VRAM Residency — Cryptographically Sealed Model Shards
//!
//! This crate enforces VRAM-only inference policy and provides sealed ModelShardHandle
//! with cryptographic integrity verification.
//!
//! ## ⚠️ CRITICAL: Shared GPU Considerations (Home/Desktop Use)
//!
//! **On home/desktop systems, the GPU is shared with user applications** (gaming, video editing,
//! 3D rendering, browser GPU acceleration, etc.). This crate handles VRAM allocation gracefully
//! but **does NOT automatically evict models** when users need the GPU.
//!
//! ### What Happens:
//!
//! 1. **User app active → Model load fails**: Returns `InsufficientVram` error (safe)
//! 2. **Model loaded → User starts app**: User's app may fail to start or run degraded (BAD UX!)
//!
//! ### Required Solution (Higher Level):
//!
//! `worker-orcd` or `pool-managerd` MUST implement:
//! - GPU process detection (via NVML or process monitoring)
//! - Automatic model eviction when user apps detected
//! - Graceful resume when user apps stop
//!
//! See: `.specs/45_shared_gpu_contention.md` for full details and implementation guide.
//!
//! ## Security Properties
//!
//! - **TIER 1 Clippy configuration** (security-critical)
//! - Private VRAM pointers (never exposed)
//! - Cryptographic seal verification (HMAC-SHA256)
//! - Bounds checking on all operations
//! - Prevents paging to host memory
//!
//! # ⚠️ AUDIT LOGGING REQUIRED
//!
//! **CRITICAL**: All VRAM operations MUST be logged to `audit-logging`:
//! ```rust,ignore
//! use audit_logging::{AuditLogger, AuditEvent};
//!
//! // ✅ VRAM sealing (security-critical)
//! audit_logger.emit(AuditEvent::VramSealed {
//!     timestamp: Utc::now(),
//!     shard_id: shard.id.clone(),
//!     gpu_device: shard.gpu_device,
//!     vram_bytes: shard.size_bytes,
//!     digest: shard.digest.clone(),
//!     worker_id: "worker-gpu-0".to_string(),
//! }).await?;
//!
//! // ✅ Seal verification
//! audit_logger.emit(AuditEvent::SealVerified {
//!     timestamp: Utc::now(),
//!     shard_id: shard.id.clone(),
//!     gpu_device: shard.gpu_device,
//!     worker_id: "worker-gpu-0".to_string(),
//! }).await?;
//!
//! // ✅ Seal verification failure
//! audit_logger.emit(AuditEvent::SealVerificationFailed {
//!     timestamp: Utc::now(),
//!     shard_id: shard.id.clone(),
//!     gpu_device: shard.gpu_device,
//!     reason: "digest_mismatch".to_string(),
//!     worker_id: "worker-gpu-0".to_string(),
//! }).await?;
//! ```
//!
//! See: `bin/shared-crates/AUDIT_LOGGING_REMINDER.md`
//!
//! ---
//!
//! # ⚠️ CRITICAL: Seal Key Management
//!
//! **DO NOT HAND-ROLL SEAL KEY HANDLING**
//!
//! For seal key loading and HMAC key derivation, use `secrets-management`:
//!
//! ```rust,ignore
//! use secrets_management::{Secret, SecretKey};
//!
//! // Load seal key from systemd credential
//! let seal_key = SecretKey::from_systemd_credential("seal_key")?;
//!
//! // Or derive from worker token (HKDF-SHA256)
//! let worker_token = Secret::load_from_file("/etc/llorch/secrets/worker-token")?;
//! let seal_key = SecretKey::derive_from_token(
//!     worker_token.expose(),
//!     b"llorch-seal-key-v1"  // Domain separation
//! )?;
//!
//! // Use seal_key.as_bytes() for HMAC operations
//! ```
//!
//! See: `bin/shared-crates/secrets-management/README.md`
//!
//! ---
//!
//! # Example
//!
//! ```rust
//! use vram_residency::{VramManager, SealedShard};
//!
//! let manager = VramManager::new();
//!
//! // Load model into VRAM
//! let shard = manager.seal_model(model_bytes, gpu_device)?;
//!
//! // Verify seal before inference
//! manager.verify_sealed(&shard)?;
//!
//! // Execute (VRAM-only guaranteed)
//! manager.execute(&shard, prompt)?;
//! ```

// Security-critical crate: TIER 1 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![deny(clippy::integer_arithmetic)]
#![deny(clippy::cast_ptr_alignment)]
#![deny(clippy::mem_forget)]
// All core functionality implemented (audit logger integration pending)
#![deny(clippy::unimplemented)]
#![warn(clippy::arithmetic_side_effects)]
#![warn(clippy::cast_lossless)]
#![warn(clippy::cast_possible_truncation)]
#![warn(clippy::cast_possible_wrap)]
#![warn(clippy::cast_precision_loss)]
#![warn(clippy::cast_sign_loss)]
#![warn(clippy::string_slice)]
#![warn(clippy::missing_errors_doc)]
#![warn(clippy::missing_panics_doc)]
#![warn(clippy::missing_safety_doc)]
#![warn(clippy::must_use_candidate)]

//
// Module structure
//

// Core types
pub mod types;
pub mod error;

// CUDA FFI (production-ready)
pub mod cuda_ffi;

// Cryptographic sealing
pub mod seal;

// VRAM allocation
pub mod allocator;

// Policy enforcement
pub mod policy;

// Input validation
pub mod validation;

// Audit logging
pub mod audit;

// Narration (observability)
pub mod narration;

//
// Public API exports
//

pub use error::{Result, VramError};
pub use types::{SealedShard, VramConfig};
pub use allocator::VramManager;

// Re-export seal functions for convenience
pub use seal::{compute_digest, compute_signature, verify_signature};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seal_model() {
        let mut manager = VramManager::new();
        let model_bytes = vec![0u8; 1_000_000]; // 1MB model

        let result = manager.seal_model(&model_bytes, 0);
        assert!(result.is_ok());

        if let Ok(shard) = result {
            assert_eq!(shard.vram_bytes, 1_000_000);
            assert!(manager.verify_sealed(&shard).is_ok());
        }
    }
}
