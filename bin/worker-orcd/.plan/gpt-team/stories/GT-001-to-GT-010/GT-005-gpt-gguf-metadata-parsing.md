# GT-005: GPT GGUF Metadata Parsing

**Team**: GPT-Gamma  
**Sprint**: Sprint 1 (HF Tokenizer)  
**Size**: M (2 days)  
**Days**: 20-21  
**Spec Ref**: M0-W-1211, M0-W-1212

---

## Story Description

Implement GGUF metadata parsing specific to GPT architecture models. Extract model configuration including layer count, embedding dimensions, attention heads, and architecture type to enable proper GPT-OSS-20B model loading and inference setup.

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

---

## Testing Strategy

### Unit Tests
- Test parsing valid GPT-OSS-20B metadata
- Test error handling for missing required keys
- Test error handling for invalid architecture type
- Test derived parameter calculations
- Test metadata validation logic

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
- GGUF Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Related Stories: GT-006, GT-007

---

**Status**: Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Detailed by Project Management Team — ready to implement 📋
