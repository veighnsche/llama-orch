//! GGUF parser primitives
//!
//! All reads are bounds-checked to prevent buffer overflows.
//! Implements security requirements from 20_security.md §3.1

use crate::error::{LoadError, Result};
use crate::validation::gguf::limits;

/// Read u32 from bytes at offset (little-endian)
///
/// # Security
/// - Bounds-checked before reading (prevents CWE-119)
/// - Uses checked arithmetic (prevents integer overflow)
/// - Returns BufferOverflow error if out of bounds
///
/// # Errors
/// - `LoadError::BufferOverflow` if offset + 4 > bytes.len()
pub fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    // Checked addition to prevent integer overflow
    let end = offset.checked_add(4)
        .ok_or(LoadError::BufferOverflow {
            offset,
            length: 4,
            available: bytes.len(),
        })?;
    
    // Bounds check
    if end > bytes.len() {
        return Err(LoadError::BufferOverflow {
            offset,
            length: 4,
            available: bytes.len(),
        });
    }
    
    // Safe to read now (bounds checked)
    Ok(u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ]))
}

/// Read u64 from bytes at offset (little-endian)
///
/// # Security
/// - Bounds-checked before reading
/// - Direct indexing safe after bounds check
pub fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let end = offset.checked_add(8)
        .ok_or(LoadError::BufferOverflow {
            offset,
            length: 8,
            available: bytes.len(),
        })?;
    
    // Safe: bounds already checked, direct indexing is safe
    Ok(u64::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
        bytes[offset + 4],
        bytes[offset + 5],
        bytes[offset + 6],
        bytes[offset + 7],
    ]))
}

/// Read string from bytes at offset (with length validation)
///
/// # Security
/// - Validates string length against MAX_STRING_LEN
/// - Bounds-checked before reading
/// - Validates before cast to prevent truncation on 32-bit
/// - Returns (string, next_offset)
pub fn read_string(bytes: &[u8], offset: usize) -> Result<(String, usize)> {
    // Read string length (u64)
    let str_len_u64 = read_u64(bytes, offset)?;
    
    // Validate BEFORE cast to prevent truncation on 32-bit systems (GGUF-003)
    if str_len_u64 > limits::MAX_STRING_LEN as u64 {
        return Err(LoadError::StringTooLong {
            length: str_len_u64 as usize,  // Safe: already validated
            max: limits::MAX_STRING_LEN,
        });
    }
    
    // Safe cast: we know it fits in usize
    let str_len = str_len_u64 as usize;
    
    // Calculate string data offset
    let str_offset = offset.checked_add(8)
        .ok_or(LoadError::BufferOverflow {
            offset,
            length: 8,
            available: bytes.len(),
        })?;
    
    // Calculate end offset
    let end_offset = str_offset.checked_add(str_len)
        .ok_or(LoadError::BufferOverflow {
            offset: str_offset,
            length: str_len,
            available: bytes.len(),
        })?;
    
    // Bounds check
    if end_offset > bytes.len() {
        return Err(LoadError::BufferOverflow {
            offset: str_offset,
            length: str_len,
            available: bytes.len(),
        });
    }
    
    // Read string bytes
    let str_bytes = &bytes[str_offset..end_offset];
    
    // Convert to UTF-8 string
    let string = String::from_utf8(str_bytes.to_vec())
        .map_err(|_| LoadError::InvalidFormat("Invalid UTF-8 in string".to_string()))?;
    
    Ok((string, end_offset))
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_read_u32_success() {
        let bytes = vec![0x01, 0x02, 0x03, 0x04];
        let value = read_u32(&bytes, 0).unwrap();
        
        assert_eq!(value, 0x04030201); // Little-endian
    }
    
    #[test]
    fn test_read_u32_out_of_bounds() {
        let bytes = vec![0x01, 0x02];
        let result = read_u32(&bytes, 0);
        
        assert!(matches!(result, Err(LoadError::BufferOverflow { .. })));
    }
}
