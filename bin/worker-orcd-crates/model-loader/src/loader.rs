//! Model loader implementation

use crate::error::{LoadError, Result};
use crate::types::LoadRequest;
use crate::validation::{hash, path, gguf};
use std::path::PathBuf;

/// Model loader
///
/// Stateless utility for loading and validating GGUF model files.
#[derive(Debug)]
pub struct ModelLoader {
    /// Allowed root directory for model files
    allowed_root: PathBuf,
}

impl ModelLoader {
    /// Create new model loader with default allowed root
    pub fn new() -> Self {
        Self {
            allowed_root: PathBuf::from("/var/lib/llorch/models"),
        }
    }
    
    /// Create model loader with custom allowed root
    pub fn with_allowed_root(allowed_root: PathBuf) -> Self {
        Self { allowed_root }
    }
    
    /// Load and validate model from filesystem
    ///
    /// # Validation Steps
    /// 1. Path validation (canonicalization, containment check)
    /// 2. File size check (< max_size)
    /// 3. File read
    /// 4. Hash verification (if expected_hash provided)
    /// 5. GGUF format validation
    ///
    /// # Security
    /// - All validation steps are fail-fast
    /// - Path traversal is prevented (TODO: needs input-validation)
    /// - Hash mismatch rejects load
    /// - GGUF parser is bounds-checked
    pub fn load_and_validate(&self, request: LoadRequest) -> Result<Vec<u8>> {
        tracing::info!(
            path = ?request.model_path,
            "Model load started"
        );
        
        // 1. Validate path (PATH-001 to PATH-008)
        let canonical_path = path::validate_path(request.model_path, &self.allowed_root)?;
        
        // 2. Check file size
        let metadata = std::fs::metadata(&canonical_path)?;
        let file_size = metadata.len() as usize;
        
        if file_size > request.max_size {
            return Err(LoadError::TooLarge {
                actual: file_size,
                max: request.max_size,
            });
        }
        
        // 3. Read file
        let model_bytes = std::fs::read(&canonical_path)?;
        
        // 4. Verify hash (HASH-001 to HASH-007)
        if let Some(expected_hash) = request.expected_hash {
            hash::verify_hash(&model_bytes, expected_hash)?;
        }
        
        // 5. Validate GGUF format (GGUF-001 to GGUF-012)
        gguf::validate_gguf(&model_bytes)?;
        
        tracing::info!(
            path = ?canonical_path,
            size = file_size,
            "Model load completed"
        );
        
        Ok(model_bytes)
    }
    /// Validate model bytes (already in memory)
    ///
    /// Used when pool-managerd sends bytes directly to worker.
    ///
    /// # Validation Steps
    /// Validate model bytes without loading from filesystem
    ///
    /// Useful for testing or when bytes are already in memory.
    pub fn validate_bytes(&self, bytes: &[u8], expected_hash: Option<&str>) -> Result<()> {
        self.validate_bytes_with_size(bytes, expected_hash, None)
    }
    
    /// Validate model bytes with optional size limit
    ///
    /// Internal helper that supports size checking.
    fn validate_bytes_with_size(&self, bytes: &[u8], expected_hash: Option<&str>, max_size: Option<usize>) -> Result<()> {
        tracing::debug!(
            size = bytes.len(),
            "Validating model bytes"
        );
        
        // 1. Check size limit if specified
        if let Some(max) = max_size {
            if bytes.len() > max {
                return Err(LoadError::TooLarge {
                    actual: bytes.len(),
                    max,
                });
            }
        }
        
        // 2. Verify hash
        if let Some(expected_hash) = expected_hash {
            hash::verify_hash(bytes, expected_hash)?;
        }
        
        // 3. Validate GGUF format
        gguf::validate_gguf(bytes)?;
        
        tracing::debug!("Model bytes validated");
        Ok(())
    }
    // TODO(Post-M0): Add metadata extraction per 30_dependencies.md §1.4
    // #[cfg(feature = "metadata-extraction")]
    // pub fn extract_metadata(&self, bytes: &[u8]) -> Result<GgufMetadata> {
    //     gguf::extract_metadata(bytes)
    // }
    
    // TODO(Post-M0): Add async variant per 30_dependencies.md §1.6
    // #[cfg(feature = "async")]
    // pub async fn load_and_validate_async(&self, request: LoadRequest<'_>) -> Result<Vec<u8>> {
    //     // Non-blocking file I/O
    //     let canonical_path = path::validate_path(request.model_path, &self.allowed_root)?;
    //     let bytes = tokio::fs::read(&canonical_path).await?;
    //     self.validate_bytes(&bytes, request.expected_hash)?;
    //     Ok(bytes)
    // }
    
    // TODO(Post-M0): Add signature verification per 30_dependencies.md §1.5
    // #[cfg(feature = "signature-verification")]
    // fn verify_signature(&self, bytes: &[u8], sig: &[u8], pubkey: &PublicKey) -> Result<()> {
    //     use ed25519_dalek::Verifier;
    //     
    //     let signature = ed25519_dalek::Signature::from_bytes(sig)
    //         .map_err(|_| LoadError::SignatureVerificationFailed)?;
    //     
    //     pubkey.verify(bytes, &signature)
    //         .map_err(|_| LoadError::SignatureVerificationFailed)?;
    //     
    //     Ok(())
    // }
}

impl Default for ModelLoader {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    
    #[test]
    fn test_validate_bytes_valid_gguf() {
        let loader = ModelLoader::new();
        
        // Valid GGUF header
        let mut bytes = vec![0x47, 0x47, 0x55, 0x46]; // "GGUF" magic
        bytes.extend_from_slice(&3u32.to_le_bytes()); // Version 3
        bytes.extend_from_slice(&1u64.to_le_bytes()); // Tensor count: 1
        bytes.extend_from_slice(&0u64.to_le_bytes()); // Metadata KV count: 0
        
        assert!(loader.validate_bytes(&bytes, None).is_ok());
    }
    
    #[test]
    fn test_validate_bytes_invalid_magic() {
        let loader = ModelLoader::new();
        let bytes = vec![0x00, 0x00, 0x00, 0x00];
        
        let result = loader.validate_bytes(&bytes, None);
        assert!(matches!(result, Err(LoadError::InvalidFormat(_))));
    }
}
