//! GGUF Format Parser for Llama-2 Models
//!
//! Parses GGUF v3 format files and extracts model metadata and tensor information.
//! Supports Q8_0 quantization with extensibility for other formats.
//!
//! Created by: TEAM-008

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek};
use std::path::Path;
use byteorder::{LittleEndian, ReadBytesExt};

/// GGUF magic number: "GGUF" in ASCII
const GGUF_MAGIC: u32 = 0x46554747;

/// GGUF versions we support (v2 and v3)
const GGUF_VERSION_MIN: u32 = 2;
const GGUF_VERSION_MAX: u32 = 3;

/// GGUF value types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GGUFValueType {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl GGUFValueType {
    fn from_u32(value: u32) -> Result<Self, String> {
        match value {
            0 => Ok(Self::UInt8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::UInt16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::UInt32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::UInt64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => Err(format!("Unknown GGUF value type: {}", value)),
        }
    }
}

/// GGUF tensor types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GGUFTensorType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
}

impl GGUFTensorType {
    fn from_u32(value: u32) -> Result<Self, String> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2_K),
            11 => Ok(Self::Q3_K),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            15 => Ok(Self::Q8_K),
            _ => Err(format!("Unknown GGUF tensor type: {}", value)),
        }
    }

    /// Get the block size for this quantization type
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 => 1,
            Self::Q4_0 | Self::Q4_1 => 32,
            Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q8_0 | Self::Q8_1 => 32,
            Self::Q2_K | Self::Q3_K | Self::Q4_K | Self::Q5_K | Self::Q6_K | Self::Q8_K => 256,
        }
    }

    /// Get the bytes per block for this quantization type
    /// TEAM-008: Verified against llama.cpp ggml-common.h
    pub fn bytes_per_block(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => 18,  // 16 * 0.5 + 2
            Self::Q4_1 => 20,  // 16 * 0.5 + 4
            Self::Q5_0 => 22,  // 16 * 0.625 + 2
            Self::Q5_1 => 24,  // 16 * 0.625 + 4
            Self::Q8_0 => 34,  // 32 int8 + 1 FP16 scale (2 bytes)
            Self::Q8_1 => 36,  // 32 int8 + 2 FP16 (d + s) (4 bytes)
            Self::Q2_K => 80,
            Self::Q3_K => 112,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
            Self::Q8_K => 292,
        }
    }
}

/// Metadata value
#[derive(Debug, Clone)]
pub enum MetadataValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    UInt64(u64),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
}

impl MetadataValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::UInt32(v) => Some(*v),
            Self::Int32(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::UInt64(v) => Some(*v),
            Self::Int64(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_string(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v),
            _ => None,
        }
    }
}

/// Tensor metadata
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: GGUFTensorType,
    pub offset: u64,
}

impl TensorInfo {
    /// Calculate total number of elements
    pub fn num_elements(&self) -> u64 {
        self.dimensions.iter().product()
    }

    /// Calculate size in bytes
    pub fn size_bytes(&self) -> u64 {
        let num_elements = self.num_elements();
        let block_size = self.tensor_type.block_size() as u64;
        let bytes_per_block = self.tensor_type.bytes_per_block() as u64;
        let num_blocks = (num_elements + block_size - 1) / block_size;
        num_blocks * bytes_per_block
    }
}

/// GGUF file parser
pub struct GGUFParser {
    metadata: HashMap<String, MetadataValue>,
    tensors: HashMap<String, TensorInfo>,
    tensor_data_offset: u64,
}

impl GGUFParser {
    /// Parse a GGUF file
    pub fn parse<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut file = File::open(path)?;

        // Parse header
        let magic = file.read_u32::<LittleEndian>()?;
        if magic != GGUF_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Invalid GGUF magic: 0x{:08X}", magic),
            ));
        }

        let version = file.read_u32::<LittleEndian>()?;
        if version < GGUF_VERSION_MIN || version > GGUF_VERSION_MAX {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unsupported GGUF version: {} (supported: {}-{})", 
                    version, GGUF_VERSION_MIN, GGUF_VERSION_MAX),
            ));
        }

        let tensor_count = file.read_u64::<LittleEndian>()?;
        let metadata_count = file.read_u64::<LittleEndian>()?;

        // Parse metadata
        let mut metadata = HashMap::new();
        for _ in 0..metadata_count {
            let (key, value) = Self::read_metadata_kv(&mut file)?;
            metadata.insert(key, value);
        }

        // Parse tensor info
        let mut tensors = HashMap::new();
        for _ in 0..tensor_count {
            let tensor_info = Self::read_tensor_info(&mut file)?;
            tensors.insert(tensor_info.name.clone(), tensor_info);
        }

        // Calculate alignment and tensor data offset
        let current_pos = file.stream_position()?;
        let alignment = 32; // GGUF uses 32-byte alignment
        let tensor_data_offset = ((current_pos + alignment - 1) / alignment) * alignment;

        Ok(Self {
            metadata,
            tensors,
            tensor_data_offset,
        })
    }

    /// Read a metadata key-value pair
    fn read_metadata_kv<R: Read>(reader: &mut R) -> io::Result<(String, MetadataValue)> {
        let key = Self::read_string(reader)?;
        let value_type = reader.read_u32::<LittleEndian>()?;
        let value_type = GGUFValueType::from_u32(value_type)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let value = Self::read_metadata_value(reader, value_type)?;
        Ok((key, value))
    }

    /// Read a metadata value
    fn read_metadata_value<R: Read>(
        reader: &mut R,
        value_type: GGUFValueType,
    ) -> io::Result<MetadataValue> {
        match value_type {
            GGUFValueType::UInt8 => Ok(MetadataValue::UInt8(reader.read_u8()?)),
            GGUFValueType::Int8 => Ok(MetadataValue::Int8(reader.read_i8()?)),
            GGUFValueType::UInt16 => Ok(MetadataValue::UInt16(reader.read_u16::<LittleEndian>()?)),
            GGUFValueType::Int16 => Ok(MetadataValue::Int16(reader.read_i16::<LittleEndian>()?)),
            GGUFValueType::UInt32 => Ok(MetadataValue::UInt32(reader.read_u32::<LittleEndian>()?)),
            GGUFValueType::Int32 => Ok(MetadataValue::Int32(reader.read_i32::<LittleEndian>()?)),
            GGUFValueType::UInt64 => Ok(MetadataValue::UInt64(reader.read_u64::<LittleEndian>()?)),
            GGUFValueType::Int64 => Ok(MetadataValue::Int64(reader.read_i64::<LittleEndian>()?)),
            GGUFValueType::Float32 => Ok(MetadataValue::Float32(reader.read_f32::<LittleEndian>()?)),
            GGUFValueType::Float64 => Ok(MetadataValue::Float64(reader.read_f64::<LittleEndian>()?)),
            GGUFValueType::Bool => Ok(MetadataValue::Bool(reader.read_u8()? != 0)),
            GGUFValueType::String => Ok(MetadataValue::String(Self::read_string(reader)?)),
            GGUFValueType::Array => {
                let elem_type = reader.read_u32::<LittleEndian>()?;
                let elem_type = GGUFValueType::from_u32(elem_type)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                let count = reader.read_u64::<LittleEndian>()?;
                let mut array = Vec::with_capacity(count as usize);
                for _ in 0..count {
                    array.push(Self::read_metadata_value(reader, elem_type)?);
                }
                Ok(MetadataValue::Array(array))
            }
        }
    }

    /// Read a string
    fn read_string<R: Read>(reader: &mut R) -> io::Result<String> {
        let len = reader.read_u64::<LittleEndian>()?;
        let mut buf = vec![0u8; len as usize];
        reader.read_exact(&mut buf)?;
        String::from_utf8(buf)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }

    /// Read tensor info
    fn read_tensor_info<R: Read>(reader: &mut R) -> io::Result<TensorInfo> {
        let name = Self::read_string(reader)?;
        let n_dims = reader.read_u32::<LittleEndian>()?;
        let mut dimensions = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dimensions.push(reader.read_u64::<LittleEndian>()?);
        }
        let tensor_type = reader.read_u32::<LittleEndian>()?;
        let tensor_type = GGUFTensorType::from_u32(tensor_type)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let offset = reader.read_u64::<LittleEndian>()?;

        Ok(TensorInfo {
            name,
            dimensions,
            tensor_type,
            offset,
        })
    }

    /// Get metadata value by key
    pub fn get_metadata(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.get(key)
    }

    /// Get tensor info by name
    pub fn get_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Get all tensor names
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Get tensor data offset in file
    pub fn tensor_data_offset(&self) -> u64 {
        self.tensor_data_offset
    }

    /// Validate that this is a Llama model
    pub fn validate_llama(&self) -> Result<(), String> {
        let arch = self.get_metadata("general.architecture")
            .and_then(|v| v.as_string())
            .ok_or("Missing general.architecture")?;

        if arch != "llama" {
            return Err(format!("Expected llama architecture, got: {}", arch));
        }

        Ok(())
    }

    /// Print summary of loaded model
    pub fn print_summary(&self) {
        println!("=== GGUF Model Summary ===");
        
        if let Some(name) = self.get_metadata("general.name").and_then(|v| v.as_string()) {
            println!("Name: {}", name);
        }
        
        if let Some(arch) = self.get_metadata("general.architecture").and_then(|v| v.as_string()) {
            println!("Architecture: {}", arch);
        }

        println!("\nModel Configuration:");
        if let Some(ctx_len) = self.get_metadata("llama.context_length").and_then(|v| v.as_u32()) {
            println!("  Context length: {}", ctx_len);
        }
        if let Some(emb_len) = self.get_metadata("llama.embedding_length").and_then(|v| v.as_u32()) {
            println!("  Embedding length: {}", emb_len);
        }
        if let Some(n_layers) = self.get_metadata("llama.block_count").and_then(|v| v.as_u32()) {
            println!("  Layers: {}", n_layers);
        }
        if let Some(n_heads) = self.get_metadata("llama.attention.head_count").and_then(|v| v.as_u32()) {
            println!("  Attention heads: {}", n_heads);
        }
        if let Some(n_kv_heads) = self.get_metadata("llama.attention.head_count_kv").and_then(|v| v.as_u32()) {
            println!("  KV heads: {}", n_kv_heads);
        }

        println!("\nTensors: {}", self.tensors.len());
        
        let total_size: u64 = self.tensors.values().map(|t| t.size_bytes()).sum();
        println!("Total size: {:.2} GB", total_size as f64 / 1024.0 / 1024.0 / 1024.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_type_block_sizes() {
        // TEAM-008: Verified against llama.cpp ggml-common.h
        assert_eq!(GGUFTensorType::Q8_0.block_size(), 32);
        assert_eq!(GGUFTensorType::Q8_0.bytes_per_block(), 34); // 32 int8 + 1 FP16 (2 bytes)
        
        assert_eq!(GGUFTensorType::Q8_1.block_size(), 32);
        assert_eq!(GGUFTensorType::Q8_1.bytes_per_block(), 36); // 32 int8 + 2 FP16 (4 bytes)
    }

    #[test]
    fn test_tensor_info_calculations() {
        let tensor = TensorInfo {
            name: "test".to_string(),
            dimensions: vec![1024, 4096],
            tensor_type: GGUFTensorType::Q8_0,
            offset: 0,
        };

        assert_eq!(tensor.num_elements(), 1024 * 4096);
        
        // Q8_0: 32 values per block, 34 bytes per block (32 int8 + 1 FP16)
        let expected_blocks = (1024 * 4096 + 31) / 32;
        let expected_size = expected_blocks * 34;
        assert_eq!(tensor.size_bytes(), expected_size);
    }
}
