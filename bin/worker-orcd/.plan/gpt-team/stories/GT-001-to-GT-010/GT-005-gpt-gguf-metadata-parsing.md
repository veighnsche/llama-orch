# GT-005: GPT GGUF Metadata Parsing

**Team**: GPT-Gamma  
**Sprint**: Sprint 1 (HF Tokenizer)  
**Size**: M (3 days) ← **+1 day for security**  
**Days**: 20-22  
**Spec Ref**: M0-W-1211, M0-W-1211a (security), M0-W-1212  
**Security Review**: auth-min Team 🎭

---

## Story Description

Implement GGUF metadata parsing specific to GPT architecture models. Extract model configuration including layer count, embedding dimensions, attention heads, and architecture type to enable proper GPT-OSS-20B model loading and inference setup.

**Security Enhancement**: Add GGUF bounds validation to prevent heap overflow vulnerabilities (CWE-119/787). Validate all tensor offsets and sizes before memory access to prevent malicious GGUF files from causing worker crashes or arbitrary code execution.

---

## Acceptance Criteria

- [ ] Parse GGUF header and extract GPT-specific metadata keys
- [ ] Extract `general.architecture` and validate it is "gpt2" or "gpt"
- [ ] Extract `gpt2.context_length` (context window size)
- [ ] Extract `gpt2.embedding_length` (hidden size/d_model)
- [ ] Extract `gpt2.block_count` (number of transformer layers)
- [ ] Extract `gpt2.attention.head_count` (number of attention heads)
- [ ] Extract `gpt2.feed_forward_length` (FFN intermediate size)
- [ ] Validate all required metadata keys are present
- [ ] Return structured GPTConfig with all parameters
- [ ] Unit tests validate metadata extraction for GPT-OSS-20B
- [ ] Error handling for missing or invalid metadata

**Security Criteria (M0-W-1211a)**:
- [ ] Validate tensor offset >= header_size + metadata_size
- [ ] Validate tensor offset < file_size
- [ ] Validate tensor offset + tensor_size <= file_size
- [ ] Check for integer overflow (offset + size doesn't wrap)
- [ ] Validate metadata string lengths < 1MB (sanity check)
- [ ] Validate array lengths < 1M elements (sanity check)
- [ ] Fuzzing tests with malformed GGUF files
- [ ] Property tests for bounds validation (1000+ random inputs)
- [ ] Edge case tests (boundary conditions, zero-size tensors)

---

## Dependencies

### Upstream (Blocks This Story)
- FT-006: FFI Interface Definition (FFI lock)
- GT-004: HF Tokenizer Conformance Tests (validated tokenizer)

### Downstream (This Story Blocks)
- GT-006: GGUF v3 Tensor Support (needs metadata parser)
- GT-007: Architecture Detection (needs GPT metadata)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/gguf/gpt_metadata.cpp` - GPT metadata parser
- `bin/worker-orcd/cuda/src/gguf/gpt_metadata.h` - GPT config struct
- `bin/worker-orcd/src/model/gpt_config.rs` - Rust GPT config struct

### Key Interfaces
```cpp
struct GPTConfig {
    std::string architecture;      // "gpt2" or "gpt"
    uint32_t context_length;       // e.g., 2048
    uint32_t embedding_length;     // e.g., 2048 (d_model)
    uint32_t block_count;          // e.g., 24 layers
    uint32_t attention_head_count; // e.g., 16 heads
    uint32_t ffn_length;           // e.g., 8192 (4 * d_model)
    uint32_t vocab_size;           // from tokenizer
};

GPTConfig parse_gpt_metadata(const GGUFMetadata& metadata);
```

```rust
#[derive(Debug, Clone)]
pub struct GPTConfig {
    pub architecture: String,
    pub context_length: u32,
    pub embedding_length: u32,
    pub block_count: u32,
    pub attention_head_count: u32,
    pub ffn_length: u32,
    pub vocab_size: u32,
}
```

### Implementation Notes
- Parse GGUF metadata key-value pairs
- Validate architecture is GPT-compatible
- Calculate derived parameters (head_dim = embedding_length / head_count)
- Fail fast if required keys missing
- Log parsed config at INFO level
- Support both "gpt2" and "gpt" architecture strings

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
- Test parsing valid GPT-OSS-20B metadata
- Test error handling for missing required keys
- Test error handling for invalid architecture type
- Test derived parameter calculations
- Test metadata validation logic

**Security Tests**:
- Test fuzzing with malformed GGUF files (offset beyond file, integer overflow, etc.)
- Test property-based validation (1000+ random tensor configurations)
- Test edge cases (offset at file boundary, zero-size tensors, max valid offset)
- Test malicious GGUF detection (crafted offsets, size mismatches)
- Test error messages don't leak sensitive information

### Integration Tests
- Test full GGUF file metadata extraction
- Test GPT-OSS-20B model config parsing
- Test config struct construction

### Manual Verification
1. Load GPT-OSS-20B GGUF file
2. Parse metadata
3. Verify all config parameters match expected values
4. Check logs show correct configuration

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing
- [ ] Integration tests passing
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.2 (Model Validation)
- Security Alert: `bin/worker-orcd/.security/SECURITY_ALERT_GGUF_PARSING.md`
- GGUF Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Security Research: https://blog.huntr.com/gguf-file-format-vulnerabilities-a-guide-for-hackers
- Related Stories: GT-006, GT-007

---

**Status**: Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---

**Security Note**: This story implements critical heap overflow prevention (CWE-119/787) discovered by llama-research team. All GGUF tensor offsets and sizes MUST be validated before memory access to prevent malicious files from compromising the worker.

---

Detailed by Project Management Team — ready to implement 📋  
Security verified by auth-min Team 🎭
