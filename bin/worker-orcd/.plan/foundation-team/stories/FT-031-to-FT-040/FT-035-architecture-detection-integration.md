# FT-035: Architecture Detection Integration

**Team**: Foundation-Alpha  
**Sprint**: Sprint 6 - Adapter + Gate 3  
**Size**: M (2 days)  
**Days**: 67 - 68  
**Spec Ref**: M0-W-1212

---

## Story Description

Integrate architecture detection from GGUF metadata with adapter factory. Model loading automatically selects correct adapter based on "general.architecture" field.

---

## Acceptance Criteria

- [ ] detect_architecture() reads GGUF metadata
- [ ] Maps "llama" → Architecture::Llama
- [ ] Maps "gpt2"/"gpt" → Architecture::GPT
- [ ] Throws error for unsupported architectures
- [ ] Integration with model loading
- [ ] Unit tests with mock GGUF files
- [ ] Integration tests with real models

---

## Dependencies

**Upstream**: FT-034 (Factory pattern, Day 66)  
**Downstream**: FT-036 (Integration tests update)

---

## Technical Details

```cpp
Architecture detect_architecture(const GGUFMetadata& metadata) {
    std::string arch = metadata.get_string("general.architecture");
    if (arch == "llama") return Architecture::Llama;
    if (arch == "gpt2" || arch == "gpt") return Architecture::GPT;
    throw std::runtime_error("Unsupported architecture: " + arch);
}

// In Model constructor:
auto arch = detect_architecture(gguf_metadata);
adapter_ = create_adapter(arch);
```

---

## Testing Strategy

- Test detection with Qwen model (llama)
- Test detection with GPT model (gpt2)
- Test error on unknown architecture
- Test integration with model loading

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Tests passing
- [ ] Story marked complete

---

**Status**: 📋 Ready  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋
