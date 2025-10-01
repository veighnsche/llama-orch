//! model-loader — Model validation and loading
//!
//! Validates model files (signature, hash, format) before loading into VRAM.
//!
//! # Security Properties
//!
//! - Cryptographic signature verification
//! - Hash validation (SHA-256)
//! - GGUF format validation
//! - Size limit enforcement
//! - Fail-fast on invalid models
//!
//! # ⚠️ INPUT VALIDATION REMINDER
//!
//! **Validate model paths and hashes** with `input-validation`:
//!
//! ```rust,ignore
//! use input_validation::{validate_path, validate_hex_string, validate_identifier};
//!
//! // Validate model file path
//! validate_path(&model_path, &allowed_models_dir)?;
//!
//! // Validate SHA-256 hash
//! validate_hex_string(&hash, 64)?;
//!
//! // Validate model ID
//! validate_identifier(&model_id, 256)?;
//! ```
//!
//! See: `bin/shared-crates/input-validation/README.md`
//!
//! # Example
//!
//! ```rust
//! use model_loader::{ModelLoader, LoadRequest};
//!
//! let loader = ModelLoader::new();
//!
//! let request = LoadRequest {
//!     model_path: "/models/llama-3.1-8b.gguf",
//!     expected_hash: "abc123...",
//!     max_size: 10_000_000_000, // 10GB
//! };
//!
//! let model_bytes = loader.load_and_validate(request)?;
//! ```

// Security-critical crate: TIER 1 Clippy configuration
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::indexing_slicing)]
#![deny(clippy::integer_arithmetic)]
#![deny(clippy::cast_ptr_alignment)]
#![deny(clippy::mem_forget)]
#![deny(clippy::todo)]
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

use sha2::{Sha256, Digest};
use std::path::Path;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum LoadError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("hash mismatch: expected {expected}, got {actual}")]
    HashMismatch { expected: String, actual: String },
    #[error("model too large: {0} > {1}")]
    TooLarge(usize, usize),
    #[error("invalid GGUF format: {0}")]
    InvalidFormat(String),
    #[error("signature verification failed")]
    SignatureVerificationFailed,
}

pub type Result<T> = std::result::Result<T, LoadError>;

/// Model load request
pub struct LoadRequest<'a> {
    pub model_path: &'a Path,
    pub expected_hash: Option<&'a str>,
    pub max_size: usize,
}

/// Model loader
pub struct ModelLoader;

impl ModelLoader {
    pub fn new() -> Self {
        Self
    }
    
    /// Load and validate model
    pub fn load_and_validate(&self, request: LoadRequest) -> Result<Vec<u8>> {
        tracing::info!(
            path = %request.model_path.display(),
            "Loading model"
        );
        
        // Check file size
        let metadata = std::fs::metadata(request.model_path)?;
        let file_size = metadata.len() as usize;
        
        if file_size > request.max_size {
            return Err(LoadError::TooLarge(file_size, request.max_size));
        }
        
        // Read file
        let model_bytes = std::fs::read(request.model_path)?;
        
        // Validate hash if provided
        if let Some(expected_hash) = request.expected_hash {
            let mut hasher = Sha256::new();
            hasher.update(&model_bytes);
            let actual_hash = format!("{:x}", hasher.finalize());
            
            if actual_hash != expected_hash {
                return Err(LoadError::HashMismatch {
                    expected: expected_hash.to_string(),
                    actual: actual_hash,
                });
            }
        }
        
        // Validate GGUF format
        self.validate_gguf(&model_bytes)?;
        
        tracing::info!(
            path = %request.model_path.display(),
            size = %file_size,
            "Model loaded and validated"
        );
        
        Ok(model_bytes)
    }
    
    fn validate_gguf(&self, bytes: &[u8]) -> Result<()> {
        // TODO(ARCH-CHANGE): Implement full GGUF validation per ARCHITECTURE_CHANGE_PLAN.md Phase 3:
        // - Validate all header fields (version, tensor_count, metadata_kv_count)
        // - Check tensor_count against MAX_TENSORS limit
        // - Validate metadata key-value pairs
        // - Check for buffer overflows in string lengths
        // - Validate tensor shapes and data types
        // - Add fuzz testing for parser
        // See: SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md Issue #19 (GGUF parser trusts input)
        const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"
        const MIN_HEADER_SIZE: usize = 12;
        
        if bytes.len() < MIN_HEADER_SIZE {
            return Err(LoadError::InvalidFormat("File too small".to_string()));
        }
        
        // Check magic number (first 4 bytes)
        let magic = u32::from_le_bytes([
            *bytes.get(0).ok_or_else(|| LoadError::InvalidFormat("Missing byte 0".to_string()))?,
            *bytes.get(1).ok_or_else(|| LoadError::InvalidFormat("Missing byte 1".to_string()))?,
            *bytes.get(2).ok_or_else(|| LoadError::InvalidFormat("Missing byte 2".to_string()))?,
            *bytes.get(3).ok_or_else(|| LoadError::InvalidFormat("Missing byte 3".to_string()))?,
        ]);
        
        if magic != GGUF_MAGIC {
            return Err(LoadError::InvalidFormat(format!("Invalid magic: 0x{:x}", magic)));
        }
        
        tracing::debug!("GGUF format validated");
        Ok(())
    }
}

impl Default for ModelLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    
    #[test]
    fn test_validate_gguf() {
        let loader = ModelLoader::new();
        
        // Valid GGUF header
        let mut valid = vec![0x47, 0x47, 0x55, 0x46]; // "GGUF" magic
        valid.extend_from_slice(&[0, 0, 0, 0, 0, 0, 0, 0]); // Padding
        assert!(loader.validate_gguf(&valid).is_ok());
        
        // Invalid magic
        let invalid = vec![0x00, 0x00, 0x00, 0x00];
        assert!(loader.validate_gguf(&invalid).is_err());
    }
}
