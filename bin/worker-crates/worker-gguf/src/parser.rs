//! GGUF Binary Parser
//!
//! Parses GGUF binary format according to spec:
//! https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::collections::HashMap;
use byteorder::{LittleEndian, ReadBytesExt};
use crate::{GGUFError, MetadataValue};

/// Tensor metadata from GGUF file
#[derive(Debug, Clone)]
pub struct TensorMetadata {
    pub name: String,
    pub ggml_type: u32,
    pub dimensions: Vec<u64>,
    pub offset: u64,
}

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian
const GGUF_VERSION: u32 = 3;

/// GGUF binary parser
pub struct GGUFParser {
    file: File,
}

impl GGUFParser {
    /// Create new parser for GGUF file
    pub fn new(path: &str) -> Result<Self, GGUFError> {
        let file = File::open(path)?;
        Ok(Self { file })
    }
    
    /// Parse GGUF file and extract metadata
    pub fn parse(&mut self) -> Result<HashMap<String, MetadataValue>, GGUFError> {
        // Parse header
        let magic = self.file.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            return Err(GGUFError::InvalidMagic);
        }
        
        let version = self.file.read_u32::<LittleEndian>()?;
        if version != GGUF_VERSION {
            return Err(GGUFError::UnsupportedVersion(version));
        }
        
        let _tensor_count = self.file.read_u64::<LittleEndian>()?;
        let metadata_count = self.file.read_u64::<LittleEndian>()?;
        
        // Parse metadata key-value pairs
        let mut metadata = HashMap::new();
        for _ in 0..metadata_count {
            let (key, value) = self.parse_metadata_kv()?;
            metadata.insert(key, value);
        }
        
        Ok(metadata)
    }
    
    /// Parse GGUF file and extract tensor metadata
    pub fn parse_tensors(&mut self) -> Result<Vec<TensorMetadata>, GGUFError> {
        // Parse header
        let magic = self.file.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            return Err(GGUFError::InvalidMagic);
        }
        
        let version = self.file.read_u32::<LittleEndian>()?;
        if version != GGUF_VERSION {
            return Err(GGUFError::UnsupportedVersion(version));
        }
        
        let tensor_count = self.file.read_u64::<LittleEndian>()?;
        let metadata_count = self.file.read_u64::<LittleEndian>()?;
        
        // Skip metadata section
        for _ in 0..metadata_count {
            let _key = self.read_string()?;
            self.skip_metadata_value()?;
        }
        
        // Parse tensor info section
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        let mut alignment_offset = 0u64;
        
        for _ in 0..tensor_count {
            let name = self.read_string()?;
            let n_dims = self.file.read_u32::<LittleEndian>()?;
            
            let mut dimensions = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dimensions.push(self.file.read_u64::<LittleEndian>()?);
            }
            
            let ggml_type = self.file.read_u32::<LittleEndian>()?;
            let offset = self.file.read_u64::<LittleEndian>()?;
            
            tensors.push(TensorMetadata {
                name,
                ggml_type,
                dimensions,
                offset,
            });
        }
        
        // Calculate alignment - tensor data starts after tensor info section
        // aligned to 32 bytes
        let current_pos = self.file.stream_position()?;
        let alignment = 32u64;
        let aligned_pos = ((current_pos + alignment - 1) / alignment) * alignment;
        
        // Adjust all tensor offsets to be absolute file positions
        for tensor in &mut tensors {
            tensor.offset += aligned_pos;
        }
        
        Ok(tensors)
    }
    
    /// Skip a metadata value (for tensor parsing)
    fn skip_metadata_value(&mut self) -> Result<(), GGUFError> {
        let value_type = self.file.read_u32::<LittleEndian>()?;
        
        match value_type {
            0 | 1 => self.file.seek(SeekFrom::Current(1))?,
            2 | 3 => self.file.seek(SeekFrom::Current(2))?,
            4 | 5 | 6 => self.file.seek(SeekFrom::Current(4))?,
            7 => self.file.seek(SeekFrom::Current(1))?,
            8 => { self.read_string()?; 0 },
            9 => {
                let elem_type = self.file.read_u32::<LittleEndian>()?;
                let count = self.file.read_u64::<LittleEndian>()?;
                
                match elem_type {
                    8 => {
                        for _ in 0..count {
                            self.read_string()?;
                        }
                        0
                    }
                    0 | 1 => self.file.seek(SeekFrom::Current(count as i64))?,
                    2 | 3 => self.file.seek(SeekFrom::Current((count * 2) as i64))?,
                    4 | 5 | 6 => self.file.seek(SeekFrom::Current((count * 4) as i64))?,
                    10 | 11 | 12 => self.file.seek(SeekFrom::Current((count * 8) as i64))?,
                    _ => return Err(GGUFError::InvalidValue(format!("Unknown array elem type: {}", elem_type))),
                }
            }
            10 | 11 | 12 => self.file.seek(SeekFrom::Current(8))?,
            _ => return Err(GGUFError::InvalidValue(format!("Unknown value type: {}", value_type))),
        };
        
        Ok(())
    }
    
    /// Parse a single metadata key-value pair
    fn parse_metadata_kv(&mut self) -> Result<(String, MetadataValue), GGUFError> {
        // Parse key (string)
        let key = self.read_string()?;
        
        // Parse value type
        let value_type = self.file.read_u32::<LittleEndian>()?;
        
        // Parse value based on type
        let value = match value_type {
            0 => MetadataValue::Int(self.file.read_u8()? as i64),
            1 => MetadataValue::Int(self.file.read_i8()? as i64),
            2 => MetadataValue::Int(self.file.read_u16::<LittleEndian>()? as i64),
            3 => MetadataValue::Int(self.file.read_i16::<LittleEndian>()? as i64),
            4 => MetadataValue::Int(self.file.read_u32::<LittleEndian>()? as i64),
            5 => MetadataValue::Int(self.file.read_i32::<LittleEndian>()? as i64),
            6 => MetadataValue::Float(self.file.read_f32::<LittleEndian>()? as f64),
            7 => MetadataValue::Bool(self.file.read_u8()? != 0),
            8 => MetadataValue::String(self.read_string()?),
            9 => self.parse_array()?,
            10 => MetadataValue::Int(self.file.read_u64::<LittleEndian>()? as i64),
            11 => MetadataValue::Int(self.file.read_i64::<LittleEndian>()?),
            12 => MetadataValue::Float(self.file.read_f64::<LittleEndian>()?),
            _ => return Err(GGUFError::InvalidValue(format!("Unknown type {} for key {}", value_type, key))),
        };
        
        Ok((key, value))
    }
    
    /// Read a string from the file
    fn read_string(&mut self) -> Result<String, GGUFError> {
        let len = self.file.read_u64::<LittleEndian>()? as usize;
        
        // Security: limit string length to prevent memory exhaustion
        if len > 1024 * 1024 {
            return Err(GGUFError::InvalidValue(format!("String too long: {} bytes", len)));
        }
        
        let mut buf = vec![0u8; len];
        self.file.read_exact(&mut buf)?;
        
        String::from_utf8(buf)
            .map_err(|_| GGUFError::InvalidValue("Invalid UTF-8 in string".to_string()))
    }
    
    /// Parse an array value
    fn parse_array(&mut self) -> Result<MetadataValue, GGUFError> {
        let elem_type = self.file.read_u32::<LittleEndian>()?;
        let count = self.file.read_u64::<LittleEndian>()?;
        
        // Security: limit array size
        if count > 10_000_000 {
            return Err(GGUFError::InvalidValue(format!("Array too large: {} elements", count)));
        }
        
        // Parse array contents based on element type
        match elem_type {
            8 => {
                // String array - read all strings (for tokenizer vocab/merges)
                let mut strings = Vec::with_capacity(count as usize);
                for _ in 0..count {
                    strings.push(self.read_string()?);
                }
                Ok(MetadataValue::StringArray(strings))
            }
            0 | 1 => {
                // u8/i8 array - skip bytes
                self.file.seek(SeekFrom::Current(count as i64))?;
                Ok(MetadataValue::Array { elem_type, count })
            }
            2 | 3 => {
                // u16/i16 array - skip 2 bytes each
                self.file.seek(SeekFrom::Current((count * 2) as i64))?;
                Ok(MetadataValue::Array { elem_type, count })
            }
            4 | 5 | 6 => {
                // u32/i32/f32 array - skip 4 bytes each
                self.file.seek(SeekFrom::Current((count * 4) as i64))?;
                Ok(MetadataValue::Array { elem_type, count })
            }
            10 | 11 | 12 => {
                // u64/i64/f64 array - skip 8 bytes each
                self.file.seek(SeekFrom::Current((count * 8) as i64))?;
                Ok(MetadataValue::Array { elem_type, count })
            }
            _ => {
                Err(GGUFError::InvalidValue(format!("Unsupported array element type: {}", elem_type)))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Write, Cursor};
    use byteorder::WriteBytesExt;
    
    #[test]
    fn test_gguf_magic() {
        assert_eq!(GGUF_MAGIC, 0x46554747);
    }
    
    // Note: String reading is tested via integration tests with real GGUF files
    
    #[test]
    fn test_invalid_magic() {
        // Create temp file with invalid magic
        use std::io::Write;
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        tmpfile.write_u32::<LittleEndian>(0x00000000).unwrap();
        tmpfile.flush().unwrap();
        
        let mut parser = GGUFParser::new(tmpfile.path().to_str().unwrap()).unwrap();
        let result = parser.parse();
        
        assert!(matches!(result, Err(GGUFError::InvalidMagic)));
    }
    
    #[test]
    fn test_unsupported_version() {
        use std::io::Write;
        let mut tmpfile = tempfile::NamedTempFile::new().unwrap();
        tmpfile.write_u32::<LittleEndian>(GGUF_MAGIC).unwrap();
        tmpfile.write_u32::<LittleEndian>(999).unwrap(); // Invalid version
        tmpfile.flush().unwrap();
        
        let mut parser = GGUFParser::new(tmpfile.path().to_str().unwrap()).unwrap();
        let result = parser.parse();
        
        assert!(matches!(result, Err(GGUFError::UnsupportedVersion(999))));
    }
}
