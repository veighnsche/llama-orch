# GT-007: Architecture Detection (GPT)

**Team**: GPT-Gamma  
**Sprint**: Sprint 1 (HF Tokenizer)  
**Size**: S (1 day)  
**Days**: 24  
**Spec Ref**: M0-W-1212

---

## Story Description

Implement architecture detection logic to identify GPT-style models from GGUF metadata and select the appropriate GPTInferenceAdapter. This enables the worker to automatically route GPT-OSS-20B to the correct inference pipeline with GPT-specific kernels.

---

## Acceptance Criteria

- [ ] Detect "gpt2" or "gpt" architecture from GGUF metadata
- [ ] Return Architecture::GPT enum variant for GPT models
- [ ] Validate GPT-specific metadata keys are present
- [ ] Fail fast with clear error for unsupported architectures
- [ ] Unit tests validate GPT architecture detection
- [ ] Unit tests validate error handling for unknown architectures
- [ ] Integration test detects GPT-OSS-20B correctly
- [ ] Log detected architecture at INFO level
- [ ] Documentation updated with supported architectures

---

## Dependencies

### Upstream (Blocks This Story)
- GT-005: GPT GGUF Metadata Parsing (needs metadata parser)
- GT-006: GGUF v3 Tensor Support (needs tensor info)

### Downstream (This Story Blocks)
- GT-039: GPTInferenceAdapter (needs architecture detection)
- FT-035: Architecture Detection Integration (Foundation team integration)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/cuda/src/model/architecture.cpp` - Architecture detection
- `bin/worker-orcd/cuda/src/model/architecture.h` - Architecture enum
- `bin/worker-orcd/src/model/architecture.rs` - Rust architecture enum

### Key Interfaces
```cpp
enum class Architecture {
    Llama,  // Qwen, Phi-3
    GPT,    // GPT-OSS-20B
};

Architecture detect_architecture(const GGUFMetadata& metadata) {
    std::string arch = metadata.get_string("general.architecture");
    
    if (arch == "llama") {
        return Architecture::Llama;
    }
    if (arch == "gpt2" || arch == "gpt") {
        return Architecture::GPT;
    }
    
    throw std::runtime_error("Unsupported architecture: " + arch);
}
```

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Architecture {
    Llama,  // RoPE, GQA, RMSNorm, SwiGLU
    GPT,    // Absolute pos, MHA, LayerNorm, GELU
}

impl Architecture {
    pub fn from_gguf_metadata(arch_str: &str) -> Result<Self, ArchitectureError> {
        match arch_str {
            "llama" => Ok(Architecture::Llama),
            "gpt2" | "gpt" => Ok(Architecture::GPT),
            _ => Err(ArchitectureError::Unsupported(arch_str.to_string())),
        }
    }
}
```

### Implementation Notes
- Read `general.architecture` key from GGUF metadata
- Support both "gpt2" and "gpt" strings for GPT models
- Validate required metadata keys exist for detected architecture
- Fail fast with descriptive error for unsupported architectures
- Log detected architecture with model name
- Enable adapter selection based on architecture

---

## Testing Strategy

### Unit Tests
- Test GPT architecture detection from "gpt2" string
- Test GPT architecture detection from "gpt" string
- Test Llama architecture detection (for completeness)
- Test error handling for unknown architecture
- Test error handling for missing metadata key

### Integration Tests
- Test GPT-OSS-20B architecture detection
- Test Qwen2.5 architecture detection (Llama)
- Test full model loading with architecture routing

### Manual Verification
1. Load GPT-OSS-20B GGUF file
2. Verify architecture detected as GPT
3. Check logs show correct architecture
4. Verify appropriate adapter selected

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

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.2 (Architecture Detection)
- Related Stories: GT-039 (GPT adapter), FT-033 (adapter interface)
- Gap Analysis: `M0_ARCHITECTURAL_GAP_ANALYSIS.md` (Gap 1)

---

**Status**: Ready for execution  
**Owner**: GPT-Gamma  
**Created**: 2025-10-04

---
Detailed by Project Management Team — ready to implement 📋
