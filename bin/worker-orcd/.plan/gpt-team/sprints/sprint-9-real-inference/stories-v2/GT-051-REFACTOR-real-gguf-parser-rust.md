# GT-051-REFACTOR: Real GGUF Parser in Rust

**Story ID**: GT-051-REFACTOR  
**Title**: Implement Real GGUF Binary Parser in worker-gguf  
**Size**: L (Large)  
**Estimate**: 8-10 hours  
**Priority**: P0 (CRITICAL - Blocks everything)  
**Dependencies**: None  
**Blocks**: GT-052, GT-053, GT-054, GT-055, GT-056

---

## User Story

**As a** Rust-first worker implementation  
**I want** a real GGUF binary parser in worker-gguf  
**So that** I can parse model metadata without C++ duplication

---

## Context

Currently `worker-gguf` has a **stub implementation** that detects architecture from filename:

```rust
pub fn from_file(path: &str) -> Result<Self, GGUFError> {
    // TODO: Implement actual GGUF parsing
    // For now, return stub metadata based on filename
    let metadata = Self::stub_metadata_from_filename(path);
    Ok(Self { metadata })
}
```

We deleted the C++ GGUF parser (~755 lines) because **RUST IS THE MAIN LANGUAGE**.

Now we need to implement the real parser in Rust.

---

## Acceptance Criteria

### GGUF Binary Parsing
- [ ] Parse GGUF magic number (0x46554747 "GGUF")
- [ ] Parse GGUF version (v3 = 3)
- [ ] Parse tensor count and metadata count
- [ ] Parse metadata key-value pairs
- [ ] Parse tensor info (name, dimensions, type, offset)
- [ ] Support all GGUF value types (uint, int, float, string, bool, array)

### Metadata Extraction
- [ ] Extract `general.architecture` (qwen2, llama, gpt2, etc.)
- [ ] Extract vocab size from tokenizer array
- [ ] Extract config values (hidden_dim, num_layers, etc.)
- [ ] Support architecture-specific metadata keys
- [ ] Handle missing optional keys gracefully

### Architecture Support
- [ ] Qwen2 architecture
- [ ] Llama architecture
- [ ] GPT2 architecture
- [ ] Phi3 architecture (future)

### Error Handling
- [ ] Invalid magic number
- [ ] Unsupported version
- [ ] Corrupted file
- [ ] Missing required keys
- [ ] Invalid value types

### Testing
- [ ] Unit tests for binary parsing
- [ ] Unit tests for metadata extraction
- [ ] Integration test with real Qwen2.5-0.5B GGUF file
- [ ] Test error cases
- [ ] Property-based tests for robustness

---

## Technical Design

### GGUF Format Reference

From GGUF spec (https://github.com/ggerganov/ggml/blob/master/docs/gguf.md):

```
GGUF File Structure:
┌─────────────────────────────────────┐
│ Magic: "GGUF" (4 bytes)             │
│ Version: u32 (4 bytes)              │
│ Tensor count: u64 (8 bytes)         │
│ Metadata count: u64 (8 bytes)       │
├─────────────────────────────────────┤
│ Metadata KV pairs (variable)        │
│   - Key: string (length + bytes)    │
│   - Value type: u32                 │
│   - Value: (type-specific)          │
├─────────────────────────────────────┤
│ Tensor info (variable)              │
│   - Name: string                    │
│   - Dimensions: u32 count + u64[]   │
│   - Type: u32 (ggml_type)           │
│   - Offset: u64                     │
├─────────────────────────────────────┤
│ Alignment padding                   │
├─────────────────────────────────────┤
│ Tensor data (variable)              │
└─────────────────────────────────────┘
```

### Value Types

```rust
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
```

### Implementation

**File**: `bin/worker-crates/worker-gguf/src/parser.rs` (NEW)

```rust
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::collections::HashMap;
use byteorder::{LittleEndian, ReadBytesExt};
use crate::{GGUFError, MetadataValue};

const GGUF_MAGIC: u32 = 0x46554747; // "GGUF"
const GGUF_VERSION: u32 = 3;

pub struct GGUFParser {
    file: File,
}

impl GGUFParser {
    pub fn new(path: &str) -> Result<Self, GGUFError> {
        let file = File::open(path)?;
        Ok(Self { file })
    }
    
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
        
        let tensor_count = self.file.read_u64::<LittleEndian>()?;
        let metadata_count = self.file.read_u64::<LittleEndian>()?;
        
        // Parse metadata
        let mut metadata = HashMap::new();
        for _ in 0..metadata_count {
            let (key, value) = self.parse_metadata_kv()?;
            metadata.insert(key, value);
        }
        
        Ok(metadata)
    }
    
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
            _ => return Err(GGUFError::InvalidValue(key.clone())),
        };
        
        Ok((key, value))
    }
    
    fn read_string(&mut self) -> Result<String, GGUFError> {
        let len = self.file.read_u64::<LittleEndian>()? as usize;
        let mut buf = vec![0u8; len];
        self.file.read_exact(&mut buf)?;
        String::from_utf8(buf).map_err(|_| GGUFError::InvalidValue("string".to_string()))
    }
    
    fn parse_array(&mut self) -> Result<MetadataValue, GGUFError> {
        let elem_type = self.file.read_u32::<LittleEndian>()?;
        let count = self.file.read_u64::<LittleEndian>()?;
        
        // For now, just store count for array length queries
        // Full array parsing can be added later if needed
        Ok(MetadataValue::Array { elem_type, count })
    }
}
```

**File**: `bin/worker-crates/worker-gguf/src/lib.rs` (UPDATE)

```rust
mod parser;
use parser::GGUFParser;

impl GGUFMetadata {
    pub fn from_file(path: &str) -> Result<Self, GGUFError> {
        let mut parser = GGUFParser::new(path)?;
        let metadata = parser.parse()?;
        Ok(Self { metadata })
    }
    
    // Remove stub_metadata_from_filename() - no longer needed!
}

// Add Array variant to MetadataValue
pub enum MetadataValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    Array { elem_type: u32, count: u64 },  // NEW
}
```

**File**: `bin/worker-crates/worker-gguf/Cargo.toml` (UPDATE)

```toml
[dependencies]
thiserror = "1.0"
serde = { version = "1.0", features = ["derive"] }
byteorder = "1.5"  # NEW - for binary parsing
```

---

## Implementation Steps

### Step 1: Add Dependencies (30 min)
1. Add `byteorder` to Cargo.toml
2. Create `parser.rs` module
3. Update lib.rs imports

### Step 2: Implement Binary Parser (3 hours)
1. Parse GGUF header (magic, version, counts)
2. Parse metadata key-value pairs
3. Handle all value types
4. Parse arrays (store count)
5. Error handling

### Step 3: Update GGUFMetadata (1 hour)
1. Replace stub with real parser
2. Update MetadataValue enum
3. Handle array length queries
4. Remove stub code

### Step 4: Architecture-Specific Keys (2 hours)
1. Support Qwen2 keys (qwen2.*)
2. Support Llama keys (llama.*)
3. Support GPT2 keys (gpt2.*)
4. Handle tokenizer.ggml.* keys
5. Fallback for missing keys

### Step 5: Testing (2-3 hours)
1. Unit tests for binary parsing
2. Unit tests for each value type
3. Integration test with real GGUF file
4. Error case tests
5. Property-based tests

### Step 6: Documentation (30 min)
1. Update README
2. Add usage examples
3. Document GGUF format

---

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_parse_magic() {
        let data = [0x47, 0x47, 0x55, 0x46]; // "GGUF"
        let mut cursor = Cursor::new(data);
        let magic = cursor.read_u32::<LittleEndian>().unwrap();
        assert_eq!(magic, GGUF_MAGIC);
    }
    
    #[test]
    fn test_parse_string() {
        let data = [
            0x05, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // len = 5
            b'h', b'e', b'l', b'l', b'o',
        ];
        let mut cursor = Cursor::new(data);
        let s = read_string(&mut cursor).unwrap();
        assert_eq!(s, "hello");
    }
    
    #[test]
    fn test_invalid_magic() {
        let data = [0x00, 0x00, 0x00, 0x00]; // Invalid
        let mut cursor = Cursor::new(data);
        let result = parse_header(&mut cursor);
        assert!(matches!(result, Err(GGUFError::InvalidMagic)));
    }
}
```

### Integration Test

```rust
#[test]
fn test_parse_real_qwen_file() {
    let path = "/home/vince/Projects/llama-orch/.test-models/qwen/qwen2.5-0.5b-instruct-q4_k_m.gguf";
    
    let metadata = GGUFMetadata::from_file(path).unwrap();
    
    assert_eq!(metadata.architecture().unwrap(), "qwen2");
    assert_eq!(metadata.vocab_size().unwrap(), 151936);
    assert_eq!(metadata.hidden_dim().unwrap(), 896);
    assert_eq!(metadata.num_layers().unwrap(), 24);
    assert_eq!(metadata.num_heads().unwrap(), 14);
    assert_eq!(metadata.num_kv_heads().unwrap(), 2);
}
```

---

## Definition of Done

- [x] Binary GGUF parser implemented
- [x] All value types supported
- [x] Metadata extraction working
- [x] Architecture detection working
- [x] Qwen2, Llama, GPT2 support
- [x] Unit tests pass (100% coverage)
- [x] Integration test with real GGUF file passes
- [x] Error handling comprehensive
- [x] Stub code removed
- [x] Documentation updated
- [x] Code reviewed and approved

---

## Dependencies

**Requires**:
- None (foundational)

**Blocks**:
- GT-052 (needs config from Rust)
- GT-053 (needs metadata)
- All other stories

---

## Time Estimate

**Optimistic**: 8 hours  
**Realistic**: 8-10 hours  
**Pessimistic**: 12 hours (if GGUF format has edge cases)

---

## Notes

### Why This Is Critical

1. **Blocks everything** - All other stories need this
2. **Eliminates duplication** - No more C++ GGUF parser
3. **Rust-first** - Honors "Rust is the main language"
4. **Reusable** - Can use for worker-aarmd (Metal)

### Risks

- ⚠️ GGUF format edge cases
- ⚠️ Large file performance
- ⚠️ Corrupted file handling

### Mitigation

- Reference llama.cpp implementation
- Add comprehensive error handling
- Test with multiple GGUF files

---

**Created by**: Project Management Team 📋  
**Assigned to**: GPT-Gamma 🤖  
**Status**: TODO  
**Priority**: P0 (CRITICAL)

---
Test opportunities identified by Testing Team 🔍
