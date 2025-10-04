# LT-001: GGUF Header Parser

**Team**: Llama-Beta  
**Sprint**: Sprint 1 - GGUF Foundation  
**Size**: M (3 days) ← **+1 day for security**  
**Days**: 15-17  
**Spec Ref**: M0-W-1211, M0-W-1211a (security)  
**Security Review**: auth-min Team 🎭

---

## Story Description

Implement GGUF file format header parser with comprehensive bounds validation to prevent heap overflow vulnerabilities. Parse magic bytes, version, tensor count, and metadata structure while enforcing strict security checks on all offsets and sizes to protect against malicious GGUF files.

**Security Enhancement**: Add GGUF bounds validation to prevent heap overflow vulnerabilities (CWE-119/787) discovered in LT-000 research. Validate all tensor offsets and sizes before memory access to prevent malicious GGUF files from causing worker crashes or arbitrary code execution.

---

## Acceptance Criteria

- [ ] Parse GGUF magic bytes (0x47475546 "GGUF") and validate
- [ ] Parse GGUF version and validate it is version 3
- [ ] Parse tensor count and validate it is reasonable (<10,000)
- [ ] Parse metadata key-value pairs structure
- [ ] Extract tensor metadata (name, dimensions, type, offset, size)
- [ ] Return structured GGUFHeader with all parsed data
- [ ] Unit tests validate header parsing for Qwen2.5-0.5B GGUF
- [ ] Error handling for invalid magic bytes, unsupported versions
- [ ] Error handling for corrupted metadata structure

**Security Criteria (M0-W-1211a)**:
- [ ] Validate tensor offset >= header_size + metadata_size
- [ ] Validate tensor offset < file_size
- [ ] Validate tensor offset + tensor_size <= file_size
- [ ] Check for integer overflow (offset + size doesn't wrap)
- [ ] Validate metadata string lengths < 1MB (sanity check)
- [ ] Validate array lengths < 1M elements (sanity check)
- [ ] Fuzzing tests with malformed GGUF files (invalid offsets, sizes, overflows)
- [ ] Property tests for bounds validation (1000+ random inputs)
- [ ] Edge case tests (boundary conditions, zero-size tensors)
- [ ] Security audit log for rejected GGUF files

---

## Dependencies

### Upstream (Blocks This Story)
- FT-006: FFI Interface Definition (FFI lock day 15)
- FT-007: Rust FFI Bindings (FFI lock day 15)

### Downstream (This Story Blocks)
- LT-002: GGUF Metadata Extraction (needs header parser)
- LT-003: Memory-Mapped I/O (needs header structure)
- LT-005: Pre-Load Validation (needs parsed header)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/gguf/header_parser.cpp` - GGUF header parser
- `bin/worker-orcd/cuda/src/gguf/header_parser.h` - Header struct definitions
- `bin/worker-orcd/src/model/gguf_header.rs` - Rust GGUF header struct
- `bin/worker-orcd/cuda/src/gguf/security.cpp` - Bounds validation functions
- `bin/worker-orcd/cuda/src/gguf/security.h` - Security validation interface

### Key Interfaces
```cpp
struct GGUFHeader {
    uint32_t magic;           // 0x47475546 "GGUF"
    uint32_t version;         // 3
    uint64_t tensor_count;    // Number of tensors
    uint64_t metadata_kv_count; // Number of metadata entries
    size_t header_size;       // Total header size in bytes
    size_t metadata_size;     // Total metadata size in bytes
    size_t data_start;        // Offset where tensor data begins
};

struct GGUFTensor {
    std::string name;
    std::vector<uint64_t> dimensions;
    uint32_t type;            // GGML tensor type
    uint64_t offset;          // Offset from data_start
    size_t size;              // Tensor size in bytes
};

GGUFHeader parse_gguf_header(const void* file_data, size_t file_size);
```

```rust
#[derive(Debug, Clone)]
pub struct GGUFHeader {
    pub magic: u32,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub header_size: usize,
    pub metadata_size: usize,
    pub data_start: usize,
}

#[derive(Debug, Clone)]
pub struct GGUFTensor {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: u32,
    pub offset: u64,
    pub size: usize,
}
```

### Implementation Notes
- Read magic bytes in little-endian format
- Validate version == 3 (reject v1, v2)
- Parse metadata key-value pairs to calculate data_start offset
- Tensor offsets are relative to data_start, not file start
- Log parsed header at INFO level
- Fail fast on any validation error

**Security Implementation**:
```cpp
bool validate_tensor_bounds(const GGUFTensor& tensor, size_t file_size, size_t data_start) {
    // Check offset is after metadata
    if (tensor.offset < data_start) {
        tracing::error!("Tensor offset before data section: {}", tensor.offset);
        return false;
    }
    
    // Check offset is within file
    if (tensor.offset >= file_size) {
        tracing::error!("Tensor offset beyond file: {} >= {}", tensor.offset, file_size);
        return false;
    }
    
    // Calculate tensor size
    size_t tensor_size = calculate_tensor_size(tensor);
    
    // Check for integer overflow (offset + size wraps around)
    if (tensor_size > SIZE_MAX - tensor.offset) {
        tracing::error!("Integer overflow: offset={} size={}", tensor.offset, tensor_size);
        return false;
    }
    
    // Check end is within file
    if (tensor.offset + tensor_size > file_size) {
        tracing::error!("Tensor extends beyond file: {}+{} > {}", 
                       tensor.offset, tensor_size, file_size);
        return false;
    }
    
    return true;
}
```

---

## Testing Strategy

### Unit Tests
- Test parsing valid Qwen2.5-0.5B GGUF header
- Test magic bytes validation (reject invalid magic)
- Test version validation (reject v1, v2, accept v3)
- Test tensor count validation (reject > 10,000)
- Test metadata parsing (key-value pairs)
- Test data_start offset calculation

**Security Tests**:
- Test fuzzing with malformed GGUF files (offset beyond file, integer overflow, etc.)
- Test property-based validation (1000+ random tensor configurations)
- Test edge cases (offset at file boundary, zero-size tensors, max valid offset)
- Test malicious GGUF detection (crafted offsets, size mismatches)
- Test error messages don't leak sensitive information

### Integration Tests
- Test full GGUF file header parsing
- Test header parsing with FFI boundary
- Test error propagation to Rust

### Manual Verification
1. Load Qwen2.5-0.5B GGUF file
2. Parse header
3. Verify magic bytes, version, tensor count
4. Check logs show correct header structure
5. Test with malicious GGUF file (should reject)

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing (10+ tests including security)
- [ ] Integration tests passing
- [ ] Security fuzzing tests passing (100+ malformed files)
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md
- [ ] Security review by auth-min team

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.2 (Model Validation)
- Security Alert: `bin/worker-orcd/.security/SECURITY_ALERT_GGUF_PARSING.md`
- GGUF Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Security Research: https://blog.huntr.com/gguf-file-format-vulnerabilities-a-guide-for-hackers
- Related Stories: LT-002, LT-003, LT-005

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

**Security Note**: This story implements critical heap overflow prevention (CWE-119/787) discovered by llama-research team in LT-000. All GGUF tensor offsets and sizes MUST be validated before memory access to prevent malicious files from compromising the worker.

---

Detailed by Project Management Team — ready to implement 📋  
Security verified by auth-min Team 🎭
