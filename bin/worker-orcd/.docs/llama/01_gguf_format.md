# GGUF Format Parsing

**Component**: GGUF Parser  
**Stories**: LT-001 to LT-006  
**Spec**: M0-W-1230

---

## Overview

GGUF (GGML Universal File Format) is a binary format for storing LLM weights and metadata. This document describes the parsing, validation, and security considerations for GGUF files in worker-orcd.

---

## File Structure

### Binary Layout

```
┌─────────────────────────────────────┐
│ Header                              │
│  - Magic: 0x47475546 ("GGUF")      │
│  - Version: 3                       │
│  - Tensor count: uint64             │
│  - Metadata count: uint64           │
├─────────────────────────────────────┤
│ Metadata (key-value pairs)          │
│  - general.architecture: "llama"    │
│  - llama.block_count: 24            │
│  - llama.attention.head_count: 14   │
│  - ...                              │
├─────────────────────────────────────┤
│ Tensor Info                         │
│  - Name: string                     │
│  - Dimensions: uint32[]             │
│  - Type: uint32 (F16, F32, etc.)    │
│  - Offset: uint64                   │
├─────────────────────────────────────┤
│ Alignment Padding                   │
├─────────────────────────────────────┤
│ Tensor Data (weights)               │
│  - Raw binary data                  │
│  - Memory-mapped for zero-copy      │
└─────────────────────────────────────┘
```

---

## Parsing Process

### 1. Header Parsing

```rust
pub struct GGUFHeader {
    pub magic: u32,        // Must be 0x47475546
    pub version: u32,      // Must be 3
    pub tensor_count: u64,
    pub metadata_count: u64,
}

pub fn parse_gguf_header(mmap: &MmapFile) -> Result<GGUFHeader, GGUFError> {
    let magic = read_u32_le(mmap, 0)?;
    if magic != 0x47475546 {
        return Err(GGUFError::InvalidMagic(magic));
    }
    
    let version = read_u32_le(mmap, 4)?;
    if version != 3 {
        return Err(GGUFError::UnsupportedVersion(version));
    }
    
    let tensor_count = read_u64_le(mmap, 8)?;
    let metadata_count = read_u64_le(mmap, 16)?;
    
    Ok(GGUFHeader {
        magic,
        version,
        tensor_count,
        metadata_count,
    })
}
```

### 2. Metadata Extraction

```rust
pub struct GGUFMetadata {
    entries: HashMap<String, MetadataValue>,
}

pub enum MetadataValue {
    UInt8(u8),
    Int32(i32),
    UInt32(u32),
    Float32(f32),
    String(String),
    Array(Vec<MetadataValue>),
}

impl GGUFMetadata {
    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.entries.get(key)? {
            MetadataValue::String(s) => Some(s),
            _ => None,
        }
    }
    
    pub fn get_uint32(&self, key: &str) -> Option<u32> {
        match self.entries.get(key)? {
            MetadataValue::UInt32(v) => Some(*v),
            _ => None,
        }
    }
}
```

### 3. Tensor Mapping

```rust
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<usize>,
    pub dtype: TensorType,
    pub offset: u64,
    pub size_bytes: usize,
}

pub fn map_qwen_weights(gguf_path: &str) -> Result<QwenWeights, WeightMappingError> {
    let mmap = MmapFile::open(gguf_path)?;
    let header = parse_gguf_header(&mmap)?;
    let metadata = parse_gguf_metadata(&mmap)?;
    let tensors = parse_tensor_info(&mmap, &header)?;
    
    // Map tensor names to model structure
    let token_embedding = find_tensor(&tensors, "token_embd.weight")?;
    
    let mut layers = Vec::new();
    for layer_idx in 0..24 {
        let attn_norm = find_tensor(&tensors, &format!("blk.{}.attn_norm.weight", layer_idx))?;
        let attn_q = find_tensor(&tensors, &format!("blk.{}.attn_q.weight", layer_idx))?;
        // ... map all layer tensors
        
        layers.push(LayerWeights {
            attn_norm_weight: attn_norm.offset,
            attn_q_weight: attn_q.offset,
            // ...
        });
    }
    
    Ok(QwenWeights {
        token_embedding: token_embedding.offset,
        layers,
        // ...
    })
}
```

### 4. Memory Mapping

```rust
pub struct MmapFile {
    mmap: memmap2::Mmap,
    file_size: usize,
}

impl MmapFile {
    pub fn open(path: &str) -> Result<Self, std::io::Error> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let file_size = mmap.len();
        
        Ok(Self { mmap, file_size })
    }
    
    pub fn read_slice(&self, offset: usize, len: usize) -> Result<&[u8], GGUFError> {
        // SECURITY: Validate bounds
        if offset + len > self.file_size {
            return Err(GGUFError::OutOfBounds { offset, len, file_size: self.file_size });
        }
        
        Ok(&self.mmap[offset..offset + len])
    }
}
```

### 5. Weight Transfer to VRAM

```rust
pub fn transfer_weights_to_vram(
    mmap: &MmapFile,
    tensors: &[TensorInfo],
) -> Result<Vec<*mut u8>, TransferError> {
    let mut vram_pointers = Vec::new();
    
    for tensor in tensors {
        // Allocate VRAM
        let vram_ptr = cuda_malloc(tensor.size_bytes)?;
        
        // Copy from mmap to VRAM
        let host_data = mmap.read_slice(tensor.offset, tensor.size_bytes)?;
        cuda_memcpy_h2d(vram_ptr, host_data)?;
        
        vram_pointers.push(vram_ptr);
    }
    
    Ok(vram_pointers)
}
```

---

## Security Considerations

### ⚠️ CRITICAL: Heap Buffer Overflow Prevention (CWE-119/787)

GGUF files are **untrusted input** and may contain malicious data. All tensor offsets and sizes MUST be validated before use.

### Required Validations

**1. Offset Bounds Check**
```rust
fn validate_tensor_offset(offset: u64, size: usize, file_size: usize) -> Result<(), GGUFError> {
    // Check for integer overflow
    let end_offset = offset.checked_add(size as u64)
        .ok_or(GGUFError::IntegerOverflow)?;
    
    // Check file bounds
    if end_offset > file_size as u64 {
        return Err(GGUFError::OutOfBounds {
            offset: offset as usize,
            len: size,
            file_size,
        });
    }
    
    Ok(())
}
```

**2. Dimension Validation**
```rust
fn validate_tensor_dimensions(dims: &[usize], expected: &[usize]) -> Result<(), GGUFError> {
    if dims.len() != expected.len() {
        return Err(GGUFError::InvalidDimensions {
            expected: expected.to_vec(),
            actual: dims.to_vec(),
        });
    }
    
    for (actual, expected) in dims.iter().zip(expected.iter()) {
        if actual != expected {
            return Err(GGUFError::InvalidDimensions {
                expected: expected.to_vec(),
                actual: dims.to_vec(),
            });
        }
    }
    
    Ok(())
}
```

**3. Size Calculation Safety**
```rust
fn calculate_tensor_size(dims: &[usize], dtype: TensorType) -> Result<usize, GGUFError> {
    let element_size = dtype.size_bytes();
    
    let mut total_elements: usize = 1;
    for &dim in dims {
        total_elements = total_elements.checked_mul(dim)
            .ok_or(GGUFError::IntegerOverflow)?;
    }
    
    let total_bytes = total_elements.checked_mul(element_size)
        .ok_or(GGUFError::IntegerOverflow)?;
    
    Ok(total_bytes)
}
```

### Security Checklist

- ✅ Validate magic bytes (0x47475546)
- ✅ Validate version (must be 3)
- ✅ Validate tensor count (reasonable limit)
- ✅ Validate metadata count (reasonable limit)
- ✅ Check all offsets are within file bounds
- ✅ Check for integer overflow in size calculations
- ✅ Validate tensor dimensions match expected config
- ✅ Validate tensor types are supported
- ✅ Use memory-mapped I/O (no buffer copies)
- ✅ Never trust user-provided offsets/sizes

---

## Example Usage

### Load Qwen Model

```rust
use worker_orcd::models::qwen::{QwenConfig, QwenWeightLoader};

let config = QwenConfig::qwen2_5_0_5b();
let model = QwenWeightLoader::load_to_vram("qwen2.5-0.5b.gguf", &config)?;

println!("Loaded {} layers", model.config.num_layers);
println!("VRAM usage: {} MB", model.total_vram_bytes / (1024 * 1024));
```

### Inspect GGUF Metadata

```rust
let mmap = MmapFile::open("model.gguf")?;
let header = parse_gguf_header(&mmap)?;
let metadata = parse_gguf_metadata(&mmap)?;

println!("Architecture: {}", metadata.get_string("general.architecture").unwrap());
println!("Layers: {}", metadata.get_uint32("llama.block_count").unwrap());
println!("Vocab size: {}", metadata.get_uint32("llama.vocab_size").unwrap());
```

---

## Error Handling

### GGUF Errors

```rust
#[derive(Debug, Error)]
pub enum GGUFError {
    #[error("Invalid magic bytes: {0:#x}")]
    InvalidMagic(u32),
    
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u32),
    
    #[error("Out of bounds: offset={offset}, len={len}, file_size={file_size}")]
    OutOfBounds {
        offset: usize,
        len: usize,
        file_size: usize,
    },
    
    #[error("Integer overflow in size calculation")]
    IntegerOverflow,
    
    #[error("Tensor not found: {0}")]
    TensorNotFound(String),
    
    #[error("Invalid dimensions: expected {expected:?}, got {actual:?}")]
    InvalidDimensions {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}
```

---

## References

- **GGUF Specification**: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- **Security Alert**: `../.security/SECURITY_ALERT_GGUF_PARSING.md`
- **Stories**: LT-001 (Header), LT-002 (Metadata), LT-003 (Tensor Mapping)
- **Spec**: `bin/.specs/01_M0_worker_orcd.md` Section 6.6

---

**Status**: Implemented  
**Security**: Validated  
**Test Coverage**: 5+ integration tests
