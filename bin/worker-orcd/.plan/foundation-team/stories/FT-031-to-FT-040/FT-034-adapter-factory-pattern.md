# FT-034: Adapter Factory Pattern

**Team**: Foundation-Alpha  
**Sprint**: Sprint 6 - Adapter + Gate 3  
**Size**: M (2 days)  
**Days**: 65 - 66  
**Spec Ref**: M0-W-1213

---

## Story Description

Implement factory pattern to create appropriate InferenceAdapter based on model architecture detected from GGUF metadata.

---

## Acceptance Criteria

- [ ] Factory function: create_adapter(Architecture arch)
- [ ] Returns unique_ptr to appropriate adapter
- [ ] Throws error for unsupported architectures
- [ ] Unit tests for factory logic
- [ ] Integration with model loading

---

## Dependencies

**Upstream**: FT-033 (Adapter interface, Day 64)  
**Downstream**: FT-035 (Architecture detection)

---

## Technical Details

```cpp
enum class Architecture {
    Llama,  // Qwen, Phi-3
    GPT,    // GPT-OSS-20B
};

std::unique_ptr<InferenceAdapter> create_adapter(Architecture arch) {
    switch (arch) {
        case Architecture::Llama:
            return std::make_unique<LlamaInferenceAdapter>();
        case Architecture::GPT:
            return std::make_unique<GPTInferenceAdapter>();
        default:
            throw std::runtime_error("Unsupported architecture");
    }
}
```

---

## Testing Strategy

- Test factory creates Llama adapter
- Test factory creates GPT adapter
- Test factory throws on unknown architecture
- Test adapter polymorphism

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

---

## 🎀 Narration Opportunities

**From**: Narration-Core Team

### Events to Narrate

1. **Adapter created by factory**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_INFERENCE_ENGINE,
       action: "adapter_create",
       target: adapter_type.to_string(),
       model_ref: Some(model_name.clone()),
       human: format!("Factory created {} adapter for model {}", adapter_type, model_name),
       ..Default::default()
   });
   ```

2. **Architecture detected**
   ```rust
   narrate_auto(NarrationFields {
       actor: ACTOR_INFERENCE_ENGINE,
       action: "architecture_detect",
       target: model_path.to_string(),
       model_ref: Some(architecture.to_string()),
       human: format!("Detected architecture: {} from {}", architecture, model_path),
       ..Default::default()
   });
   ```

**Why this matters**: Factory pattern routes models to correct adapters. Narration helps track architecture detection and adapter selection.

---
*Narration guidance added by Narration-Core Team 🎀*
