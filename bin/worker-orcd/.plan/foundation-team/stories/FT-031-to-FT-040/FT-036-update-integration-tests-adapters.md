# FT-036: Update Integration Tests for Adapters

**Team**: Foundation-Alpha  
**Sprint**: Sprint 6 - Adapter + Gate 3  
**Size**: M (2 days)  
**Days**: 69 - 70  
**Spec Ref**: M0-W-1820

---

## Story Description

Update integration tests to validate adapter pattern works correctly with both Llama and GPT models. Ensure polymorphic behavior is correct.

---

## Acceptance Criteria

- [ ] Test Llama model loads with LlamaInferenceAdapter
- [ ] Test GPT model loads with GPTInferenceAdapter
- [ ] Test adapter selection is automatic
- [ ] Test both adapters generate tokens
- [ ] Test error handling for unsupported architectures
- [ ] All existing tests still pass

---

## Dependencies

**Upstream**: FT-035 (Architecture detection, Day 68)  
**Downstream**: FT-038 (Gate 3 checkpoint)

---

## Testing Strategy

```rust
#[test]
fn test_llama_adapter_selection() {
    let model = load_model("qwen2.5-0.5b-instruct-q4_k_m.gguf");
    assert_eq!(model.adapter_name(), "LlamaInferenceAdapter");
}

#[test]
fn test_gpt_adapter_selection() {
    let model = load_model("gpt-oss-20b-mxfp4.gguf");
    assert_eq!(model.adapter_name(), "GPTInferenceAdapter");
}
```

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
