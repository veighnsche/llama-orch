//! GGUF format validation
//!
//! Implements GGUF-001 to GGUF-012 from 20_security.md
//! Prevents CWE-119 (Buffer Overflow), CWE-190 (Integer Overflow)

pub mod parser;
pub mod limits;

use crate::error::{LoadError, Result};

/// Validate GGUF format
///
/// # Security Requirements
/// - GGUF-001: Enforce maximum tensor count (default 10,000)
/// - GGUF-002: Enforce maximum file size (default 100GB)
/// - GGUF-003: Enforce maximum string length (default 64KB)
/// - GGUF-004: Enforce maximum metadata pairs (default 1,000)
/// - GGUF-005: All reads MUST be bounds-checked
/// - GGUF-010: Magic number MUST be validated (0x46554747)
/// - GGUF-011: Parser MUST fail fast on first invalid field
pub fn validate_gguf(bytes: &[u8]) -> Result<()> {
    // Validate minimum size
    if bytes.len() < limits::MIN_HEADER_SIZE {
        return Err(LoadError::InvalidFormat(
            format!("File too small: {} bytes (min {})", bytes.len(), limits::MIN_HEADER_SIZE)
        ));
    }
    
    // Validate magic number (GGUF-010)
    let magic = parser::read_u32(bytes, 0)?;
    if magic != limits::GGUF_MAGIC {
        return Err(LoadError::InvalidFormat(
            format!("Invalid magic: 0x{:x} (expected 0x{:x})", magic, limits::GGUF_MAGIC)
        ));
    }
    
    // Validate version field (GGUF-009)
    let version = parser::read_u32(bytes, 4)?;
    if version != 2 && version != 3 {
        return Err(LoadError::InvalidFormat(
            format!("Unsupported GGUF version: {} (expected 2 or 3)", version)
        ));
    }
    
    // Validate tensor count (GGUF-001)
    let tensor_count = parser::read_u64(bytes, 8)?;
    if tensor_count > limits::MAX_TENSORS as u64 {
        return Err(LoadError::TensorCountExceeded {
            count: tensor_count as usize,
            max: limits::MAX_TENSORS,
        });
    }
    
    // Validate metadata KV count (GGUF-004)
    let metadata_kv_count = parser::read_u64(bytes, 16)?;
    if metadata_kv_count > limits::MAX_METADATA_PAIRS as u64 {
        return Err(LoadError::InvalidFormat(
            format!("Metadata pair count {} exceeds maximum {}", metadata_kv_count, limits::MAX_METADATA_PAIRS)
        ));
    }
    
    tracing::debug!(
        magic = format!("0x{:x}", magic),
        version = version,
        tensor_count = tensor_count,
        metadata_kv_count = metadata_kv_count,
        "GGUF format validated"
    );
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_valid_gguf_header() {
        let mut bytes = vec![0x47, 0x47, 0x55, 0x46]; // "GGUF" magic
        bytes.extend_from_slice(&3u32.to_le_bytes()); // Version 3
        bytes.extend_from_slice(&1u64.to_le_bytes()); // Tensor count: 1
        bytes.extend_from_slice(&0u64.to_le_bytes()); // Metadata KV count: 0
        
        assert!(validate_gguf(&bytes).is_ok());
    }
    
    #[test]
    fn test_invalid_magic() {
        let bytes = vec![0x00, 0x00, 0x00, 0x00];
        
        let result = validate_gguf(&bytes);
        assert!(matches!(result, Err(LoadError::InvalidFormat(_))));
    }
    
    #[test]
    fn test_file_too_small() {
        let bytes = vec![0x47, 0x47]; // Only 2 bytes
        
        let result = validate_gguf(&bytes);
        assert!(matches!(result, Err(LoadError::InvalidFormat(_))));
    }
}
