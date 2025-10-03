# FT-033: InferenceAdapter Interface

**Team**: Foundation-Alpha  
**Sprint**: Sprint 6 - Adapter + Gate 3  
**Size**: M (2 days)  
**Days**: 63 - 64  
**Spec Ref**: M0-W-1213

---

## Story Description

Define InferenceAdapter base class to abstract architecture-specific logic (Llama vs GPT). This enables polymorphic handling of different model architectures.

---

## Acceptance Criteria

- [ ] InferenceAdapter base class defined in C++
- [ ] Virtual methods: load_weights_from_gguf, run_forward_pass, architecture_name
- [ ] LlamaInferenceAdapter stub created
- [ ] GPTInferenceAdapter stub created
- [ ] Unit tests for adapter interface
- [ ] Documentation with usage examples

---

## Dependencies

**Upstream**: FT-032 (Gate 2, Day 62)  
**Downstream**: FT-034 (Adapter factory), LT-028, GT-029

---

## Technical Details

```cpp
class InferenceAdapter {
public:
    virtual ~InferenceAdapter() = default;
    
    virtual void load_weights_from_gguf(
        const GGUFFile& gguf,
        DeviceMemory& vram_allocation
    ) = 0;
    
    virtual void run_forward_pass(
        const ModelWeights& weights,
        const DeviceMemory& input_tokens,
        DeviceMemory& output_logits,
        KVCache& kv_cache,
        cudaStream_t stream
    ) = 0;
    
    virtual const char* architecture_name() const = 0;
};

class LlamaInferenceAdapter : public InferenceAdapter { /* ... */ };
class GPTInferenceAdapter : public InferenceAdapter { /* ... */ };
```

---

## Testing Strategy

- Test adapter interface compiles
- Test virtual method dispatch
- Test stub implementations
- Test polymorphic usage

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Tests passing
- [ ] Documentation complete
- [ ] Story marked complete

---

**Status**: 📋 Ready  
**Owner**: Foundation-Alpha  
**Created**: 2025-10-04

---
Planned by Project Management Team 📋
