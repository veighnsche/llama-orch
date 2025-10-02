//! Model loader implementation

use crate::error::{LoadError, Result};
use crate::types::LoadRequest;
use crate::validation::{hash, path, gguf};
use crate::narration;
use std::path::PathBuf;
use std::time::Instant;

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
        let start = Instant::now();
        
        // Extract actor context
        let worker_id = request.worker_id.as_deref();
        let correlation_id = request.correlation_id.as_deref();
        
        let model_path_str = request.model_path.to_str().unwrap_or("<non-UTF8>");
        let max_size_gb = request.max_size as f64 / 1_000_000_000.0;
        
        // Narrate: Load start
        narration::narrate_load_start(
            model_path_str,
            max_size_gb,
            worker_id,
            correlation_id,
        );
        
        tracing::info!(
            path = ?request.model_path,
            "Model load started"
        );
        
        // 1. Validate path (PATH-001 to PATH-008)
        let canonical_path = match path::validate_path(request.model_path, &self.allowed_root) {
            Ok(p) => {
                // Narrate: Path validated
                narration::narrate_path_validated(
                    p.to_str().unwrap_or("<non-UTF8>"),
                    worker_id,
                    correlation_id,
                );
                p
            }
            Err(e) => {
                // Narrate: Path validation failed
                narration::narrate_path_validation_failed(
                    model_path_str,
                    "path_traversal",
                    worker_id,
                    correlation_id,
                );
                return Err(e);
            }
        };
        
        // 2. Check file size
        let metadata = std::fs::metadata(&canonical_path)?;
        let file_size = metadata.len() as usize;
        let file_size_gb = file_size as f64 / 1_000_000_000.0;
        
        if file_size > request.max_size {
            // Narrate: File too large
            narration::narrate_size_check_failed(
                canonical_path.to_str().unwrap_or("<non-UTF8>"),
                file_size_gb,
                max_size_gb,
                worker_id,
                correlation_id,
            );
            return Err(LoadError::TooLarge {
                actual: file_size,
                max: request.max_size,
            });
        }
        
        // Narrate: Size check passed
        narration::narrate_size_checked(
            canonical_path.to_str().unwrap_or("<non-UTF8>"),
            file_size_gb,
            max_size_gb,
            worker_id,
            correlation_id,
        );
        
        // 3. Read file
        let model_bytes = std::fs::read(&canonical_path)?;
        
        // 4. Verify hash (HASH-001 to HASH-007)
        if let Some(expected_hash) = request.expected_hash {
            let hash_start = Instant::now();
            
            // Narrate: Hash verification start
            narration::narrate_hash_verify_start(
                canonical_path.to_str().unwrap_or("<non-UTF8>"),
                worker_id,
                correlation_id,
            );
            
            match hash::verify_hash(&model_bytes, expected_hash) {
                Ok(_) => {
                    let hash_duration_ms = hash_start.elapsed().as_millis() as u64;
                    let hash_prefix = &expected_hash[..6.min(expected_hash.len())];
                    
                    // Narrate: Hash verified
                    narration::narrate_hash_verified(
                        canonical_path.to_str().unwrap_or("<non-UTF8>"),
                        hash_prefix,
                        hash_duration_ms,
                        worker_id,
                        correlation_id,
                    );
                }
                Err(LoadError::HashMismatch { expected, actual }) => {
                    let expected_prefix = &expected[..6.min(expected.len())];
                    let actual_prefix = &actual[..6.min(actual.len())];
                    
                    // Narrate: Hash mismatch
                    narration::narrate_hash_verification_failed(
                        canonical_path.to_str().unwrap_or("<non-UTF8>"),
                        expected_prefix,
                        actual_prefix,
                        worker_id,
                        correlation_id,
                    );
                    
                    return Err(LoadError::HashMismatch { expected, actual });
                }
                Err(e) => return Err(e),
            }
        }
        
        // 5. Validate GGUF format (GGUF-001 to GGUF-012)
        let gguf_start = Instant::now();
        
        // Narrate: GGUF validation start
        narration::narrate_gguf_validate_start(
            canonical_path.to_str().unwrap_or("<non-UTF8>"),
            worker_id,
            correlation_id,
        );
        
        match gguf::validate_gguf(&model_bytes) {
            Ok(_) => {
                let gguf_duration_ms = gguf_start.elapsed().as_millis() as u64;
                
                // Extract GGUF metadata for narration
                // TODO: Parse actual values from bytes
                let version = 3u32; // Placeholder
                let tensor_count = 1u64; // Placeholder
                let metadata_count = 0u64; // Placeholder
                
                // Narrate: GGUF validated
                narration::narrate_gguf_validated(
                    canonical_path.to_str().unwrap_or("<non-UTF8>"),
                    version,
                    tensor_count,
                    metadata_count,
                    gguf_duration_ms,
                    worker_id,
                    correlation_id,
                );
            }
            Err(e) => {
                // Narrate: GGUF validation failed
                // Determine error kind from error message
                let error_msg = e.to_string();
                if error_msg.contains("magic") {
                    narration::narrate_gguf_validation_failed_magic(
                        canonical_path.to_str().unwrap_or("<non-UTF8>"),
                        0x46554747,
                        0x00000000,
                        worker_id,
                        correlation_id,
                    );
                } else {
                    narration::narrate_gguf_validation_failed_bounds(
                        canonical_path.to_str().unwrap_or("<non-UTF8>"),
                        "tensors",
                        0,
                        0,
                        worker_id,
                        correlation_id,
                    );
                }
                return Err(e);
            }
        }
        
        let total_duration_ms = start.elapsed().as_millis() as u64;
        
        // Narrate: Load complete
        narration::narrate_load_complete(
            canonical_path.to_str().unwrap_or("<non-UTF8>"),
            file_size_gb,
            total_duration_ms,
            worker_id,
            correlation_id,
        );
        
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
