# LT-006: Architecture Detection (Llama)

**Team**: Llama-Beta  
**Sprint**: Sprint 1 - GGUF Foundation  
**Size**: S (1 day)  
**Days**: 26  
**Spec Ref**: M0-W-1212

---

## Story Description

Implement architecture detection for Llama-family models from GGUF metadata. Identify specific Llama variants (Qwen, Phi-3, Llama 2/3) based on metadata signatures and model dimensions to enable variant-specific optimizations and validation.

---

## Acceptance Criteria

- [ ] Parse `general.architecture` metadata key
- [ ] Validate architecture is "llama" (reject non-Llama models)
- [ ] Detect Llama variant from model dimensions and metadata
- [ ] Identify Qwen models (context_length=32768, GQA with 2 KV heads)
- [ ] Identify Phi-3 models (context_length=4096, MHA with 32 heads)
- [ ] Identify Llama 2/3 models (standard Llama architecture)
- [ ] Return structured ArchitectureInfo with variant and capabilities
- [ ] Unit tests validate detection for Qwen2.5-0.5B
- [ ] Unit tests validate detection for Phi-3
- [ ] Error handling for unknown Llama variants (warn, not fail)
- [ ] Log detected architecture at INFO level

---

## Dependencies

### Upstream (Blocks This Story)
- LT-002: GGUF Metadata Extraction (needs metadata)

### Downstream (This Story Blocks)
- LT-033: LlamaInferenceAdapter (needs architecture info)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/model/arch_detect.cpp` - Architecture detection
- `bin/worker-orcd/cuda/src/model/arch_detect.h` - Architecture info struct
- `bin/worker-orcd/src/model/arch_info.rs` - Rust architecture info

### Key Interfaces
```cpp
enum class LlamaVariant {
    Qwen,        // Qwen 2.5 series
    Phi3,        // Microsoft Phi-3
    Llama2,      // Meta Llama 2
    Llama3,      // Meta Llama 3
    Unknown,     // Unknown Llama variant
};

struct ArchitectureInfo {
    std::string architecture;  // "llama"
    LlamaVariant variant;
    bool supports_gqa;         // Grouped Query Attention
    bool supports_mha;         // Multi-Head Attention
    uint32_t kv_heads;         // Number of KV heads (GQA)
    std::string model_name;    // e.g., "Qwen2.5-0.5B", "Phi-3-mini"
};

class ArchitectureDetector {
public:
    // Detect Llama variant from config
    static ArchitectureInfo detect(const LlamaConfig& config);
    
private:
    static LlamaVariant detect_variant(const LlamaConfig& config);
    static std::string infer_model_name(const LlamaConfig& config, LlamaVariant variant);
};
```

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum LlamaVariant {
    Qwen,
    Phi3,
    Llama2,
    Llama3,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct ArchitectureInfo {
    pub architecture: String,
    pub variant: LlamaVariant,
    pub supports_gqa: bool,
    pub supports_mha: bool,
    pub kv_heads: u32,
    pub model_name: String,
}
```

### Implementation Notes
- Validate `general.architecture == "llama"` first
- Detect variant from metadata signatures:
  - **Qwen**: `context_length=32768`, `attention_head_count_kv=2` (GQA)
  - **Phi-3**: `context_length=4096`, `attention_head_count_kv=32` (MHA)
  - **Llama 2**: `context_length=4096`, `attention_head_count_kv < attention_head_count` (GQA)
  - **Llama 3**: `context_length=8192`, similar to Llama 2
- Set `supports_gqa = (kv_heads < attention_heads)`
- Set `supports_mha = (kv_heads == attention_heads)`
- Infer model name from dimensions (e.g., "Qwen2.5-0.5B" from 896 embedding)
- Log detected variant at INFO level
- Warn if variant is Unknown (but don't fail)

**Detection Logic**:
```cpp
LlamaVariant detect_variant(const LlamaConfig& config) {
    // Qwen: 32K context + GQA with 2 KV heads
    if (config.context_length == 32768 && config.attention_head_count_kv == 2) {
        return LlamaVariant::Qwen;
    }
    
    // Phi-3: 4K context + MHA (KV heads == attention heads)
    if (config.context_length == 4096 && 
        config.attention_head_count_kv == config.attention_head_count) {
        return LlamaVariant::Phi3;
    }
    
    // Llama 3: 8K context + GQA
    if (config.context_length == 8192 && 
        config.attention_head_count_kv < config.attention_head_count) {
        return LlamaVariant::Llama3;
    }
    
    // Llama 2: 4K context + GQA
    if (config.context_length == 4096 && 
        config.attention_head_count_kv < config.attention_head_count) {
        return LlamaVariant::Llama2;
    }
    
    return LlamaVariant::Unknown;
}
```

---

## Testing Strategy

### Unit Tests
- Test Qwen2.5-0.5B detection (variant=Qwen, GQA, 2 KV heads)
- Test Phi-3 detection (variant=Phi3, MHA, 32 KV heads)
- Test Llama 2 detection (variant=Llama2, GQA)
- Test Llama 3 detection (variant=Llama3, GQA)
- Test unknown variant handling (warn, not fail)
- Test architecture validation (reject non-llama)
- Test model name inference

### Integration Tests
- Test detection with real Qwen2.5-0.5B GGUF
- Test detection with real Phi-3 GGUF
- Test ArchitectureInfo struct construction

### Manual Verification
1. Load Qwen2.5-0.5B GGUF
2. Detect architecture
3. Verify variant=Qwen, supports_gqa=true, kv_heads=2
4. Check logs show "Detected Qwen2.5-0.5B (Llama variant)"

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing (7+ tests)
- [ ] Integration tests passing
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.2 (Model Validation)
- Qwen2.5 Architecture: https://qwenlm.github.io/blog/qwen2.5/
- Phi-3 Architecture: https://arxiv.org/abs/2404.14219
- Related Stories: LT-002, LT-033

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋
