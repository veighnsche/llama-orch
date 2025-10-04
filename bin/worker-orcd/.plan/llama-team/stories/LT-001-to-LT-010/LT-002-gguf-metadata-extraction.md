# LT-002: GGUF Metadata Extraction (Llama)

**Team**: Llama-Beta  
**Sprint**: Sprint 1 - GGUF Foundation  
**Size**: M (2 days)  
**Days**: 18-19  
**Spec Ref**: M0-W-1211, M0-W-1212

---

## Story Description

Extract Llama-specific metadata from GGUF files to configure model architecture. Parse model configuration including layer count, embedding dimensions, attention parameters, and architecture type to enable proper Qwen2.5-0.5B and Phi-3 model loading.

---

## Acceptance Criteria

- [ ] Parse GGUF metadata and extract Llama-specific keys
- [ ] Extract `general.architecture` and validate it is "llama"
- [ ] Extract `llama.context_length` (context window size)
- [ ] Extract `llama.embedding_length` (hidden size/d_model)
- [ ] Extract `llama.block_count` (number of transformer layers)
- [ ] Extract `llama.attention.head_count` (number of attention heads)
- [ ] Extract `llama.attention.head_count_kv` (KV heads for GQA)
- [ ] Extract `llama.feed_forward_length` (FFN intermediate size)
- [ ] Extract `llama.rope.dimension_count` (RoPE dimensions)
- [ ] Extract `llama.rope.freq_base` (RoPE frequency base, default 10000.0)
- [ ] Validate all required metadata keys are present
- [ ] Calculate derived parameters (head_dim = embedding_length / head_count)
- [ ] Return structured LlamaConfig with all parameters
- [ ] Unit tests validate metadata extraction for Qwen2.5-0.5B
- [ ] Unit tests validate metadata extraction for Phi-3
- [ ] Error handling for missing or invalid metadata

---

## Dependencies

### Upstream (Blocks This Story)
- LT-001: GGUF Header Parser (needs header structure)

### Downstream (This Story Blocks)
- LT-006: Architecture Detection (needs metadata)
- LT-022: Qwen Weight Mapping (needs config)
- LT-029: Phi-3 Metadata Analysis (needs parser)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/gguf/llama_metadata.cpp` - Llama metadata parser
- `bin/worker-orcd/cuda/src/gguf/llama_metadata.h` - Llama config struct
- `bin/worker-orcd/src/model/llama_config.rs` - Rust Llama config struct

### Key Interfaces
```cpp
struct LlamaConfig {
    std::string architecture;      // "llama"
    uint32_t context_length;       // e.g., 32768 (Qwen), 4096 (Phi-3)
    uint32_t embedding_length;     // e.g., 896 (Qwen), 3072 (Phi-3)
    uint32_t block_count;          // e.g., 24 (Qwen), 32 (Phi-3)
    uint32_t attention_head_count; // e.g., 14 (Qwen), 32 (Phi-3)
    uint32_t attention_head_count_kv; // e.g., 2 (Qwen GQA), 32 (Phi-3 MHA)
    uint32_t ffn_length;           // e.g., 4864 (Qwen), 8192 (Phi-3)
    uint32_t rope_dimension_count; // e.g., 64
    float rope_freq_base;          // e.g., 10000.0 or 1000000.0
    uint32_t vocab_size;           // from tokenizer
    
    // Derived parameters
    uint32_t head_dim;             // embedding_length / attention_head_count
    uint32_t kv_head_dim;          // embedding_length / attention_head_count_kv
};

LlamaConfig parse_llama_metadata(const GGUFMetadata& metadata);
```

```rust
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    pub architecture: String,
    pub context_length: u32,
    pub embedding_length: u32,
    pub block_count: u32,
    pub attention_head_count: u32,
    pub attention_head_count_kv: u32,
    pub ffn_length: u32,
    pub rope_dimension_count: u32,
    pub rope_freq_base: f32,
    pub vocab_size: u32,
    pub head_dim: u32,
    pub kv_head_dim: u32,
}
```

### Implementation Notes
- Parse GGUF metadata key-value pairs
- Validate architecture is "llama"
- Calculate derived parameters (head_dim, kv_head_dim)
- Default rope_freq_base to 10000.0 if not present
- Fail fast if required keys missing
- Log parsed config at INFO level
- Support both Qwen and Phi-3 metadata variations

---

## Testing Strategy

### Unit Tests
- Test parsing valid Qwen2.5-0.5B metadata
- Test parsing valid Phi-3 metadata
- Test error handling for missing required keys
- Test error handling for invalid architecture type
- Test derived parameter calculations (head_dim, kv_head_dim)
- Test default rope_freq_base value
- Test metadata validation logic

### Integration Tests
- Test full GGUF file metadata extraction
- Test Qwen2.5-0.5B model config parsing
- Test Phi-3 model config parsing
- Test config struct construction

### Manual Verification
1. Load Qwen2.5-0.5B GGUF file
2. Parse metadata
3. Verify all config parameters match expected values:
   - context_length: 32768
   - embedding_length: 896
   - block_count: 24
   - attention_head_count: 14
   - attention_head_count_kv: 2 (GQA)
4. Check logs show correct configuration

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing (8+ tests)
- [ ] Integration tests passing
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.2 (Model Validation)
- GGUF Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Qwen2.5 Model Card: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF
- Phi-3 Model Card: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf
- Related Stories: LT-001, LT-006, LT-022, LT-029

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋
