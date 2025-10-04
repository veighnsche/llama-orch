# LT-034: Gate 3 Participation

**Team**: Llama-Beta  
**Sprint**: Sprint 6 - Phi-3 + Adapter  
**Size**: S (1 day)  
**Days**: 77  
**Spec Ref**: Gate 3

---

## Story Description

Participate in Gate 3 validation to verify LlamaInferenceAdapter is complete and both Qwen and Phi-3 models work through unified adapter interface. Validate adapter pattern integration with Foundation-Alpha's adapter registry.

---

## Acceptance Criteria

- [ ] LlamaInferenceAdapter implements InferenceAdapter trait
- [ ] Qwen2.5-0.5B works via adapter interface
- [ ] Phi-3-mini-4k works via adapter interface
- [ ] Adapter supports model selection by model_ref
- [ ] Adapter integrates with adapter registry
- [ ] Both models generate coherent output via adapter
- [ ] Adapter supports load/unload lifecycle
- [ ] Integration tests pass for both models
- [ ] Gate 3 validation checklist complete
- [ ] No blocking issues for final integration
- [ ] Documentation updated with adapter usage
- [ ] Sign-off from Foundation-Alpha team

---

## Dependencies

### Upstream (Blocks This Story)
- LT-033: LlamaInferenceAdapter Implementation (needs adapter)
- FT-028: Adapter Registry (needs registry)
- FT-029: Gate 3 Checkpoint (needs gate validation)

### Downstream (This Story Blocks)
- LT-035: Llama Integration Test Suite (needs validated adapter)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/.plan/llama-team/integration-gates/gate-3-adapter-complete.md` - Gate 3 report
- `bin/worker-orcd/tests/integration/gate3_llama_adapter.rs` - Gate 3 validation tests

### Gate 3 Validation Checklist

**Adapter Interface**:
- [ ] LlamaInferenceAdapter implements InferenceAdapter trait
- [ ] All trait methods implemented correctly
- [ ] load() works for both Qwen and Phi-3
- [ ] unload() cleans up resources correctly
- [ ] encode() works for both models
- [ ] decode() works for both models
- [ ] prefill() works for both models
- [ ] decode_token() works for both models
- [ ] model_type() returns correct type
- [ ] context_length() returns correct value
- [ ] vocab_size() returns correct value

**Model Support**:
- [ ] Qwen2.5-0.5B loads via adapter
- [ ] Phi-3-mini-4k loads via adapter
- [ ] Both models generate coherent text
- [ ] Model selection by model_ref works
- [ ] Unsupported models rejected gracefully

**Adapter Registry Integration**:
- [ ] Adapter registers with registry
- [ ] Registry can create Llama adapter
- [ ] Registry can lookup adapter by model_type
- [ ] Multiple adapters can coexist

**Quality**:
- [ ] Generated text quality matches direct implementation
- [ ] No performance regression from adapter layer
- [ ] Error handling works correctly
- [ ] Logging works (tracing, not printf)

**Lifecycle**:
- [ ] Load → Generate → Unload works
- [ ] Multiple load/unload cycles work
- [ ] No memory leaks
- [ ] VRAM cleanup on unload

### Gate 3 Validation Test
```rust
#[test]
fn gate3_llama_adapter_qwen() {
    // 1. Create Qwen adapter
    let mut adapter = LlamaInferenceAdapter::from_model_ref("qwen2.5-0.5b-instruct").unwrap();
    assert_eq!(adapter.model_type(), "llama/qwen");
    
    // 2. Load model
    adapter.load(Path::new("qwen2.5-0.5b.gguf")).unwrap();
    assert_eq!(adapter.context_length(), 32768);
    assert!(adapter.vocab_size() > 150000);
    
    // 3. Generate text
    let prompt = "Write a haiku about autumn leaves";
    let input_ids = adapter.encode(prompt).unwrap();
    let prefill_output = adapter.prefill(&input_ids).unwrap();
    
    let mut generated_ids = vec![];
    let mut current_token = *prefill_output.last().unwrap();
    
    for _ in 0..30 {
        current_token = adapter.decode_token(current_token).unwrap();
        generated_ids.push(current_token);
        
        if current_token == adapter.encoder.as_ref().unwrap().eos_token_id() {
            break;
        }
    }
    
    let output = adapter.decode(&generated_ids).unwrap();
    
    // 4. Validate output
    assert!(!output.is_empty());
    assert!(output.len() > 10);
    
    println!("Qwen output: {}", output);
    
    // 5. Unload
    adapter.unload().unwrap();
}

#[test]
fn gate3_llama_adapter_phi3() {
    // 1. Create Phi-3 adapter
    let mut adapter = LlamaInferenceAdapter::from_model_ref("phi-3-mini-4k-instruct").unwrap();
    assert_eq!(adapter.model_type(), "llama/phi3");
    
    // 2. Load model
    adapter.load(Path::new("phi-3-mini-4k.gguf")).unwrap();
    assert_eq!(adapter.context_length(), 4096);
    
    // 3. Generate text
    let prompt = "The quick brown fox";
    let input_ids = adapter.encode(prompt).unwrap();
    let prefill_output = adapter.prefill(&input_ids).unwrap();
    
    let mut generated_ids = vec![];
    let mut current_token = *prefill_output.last().unwrap();
    
    for _ in 0..20 {
        current_token = adapter.decode_token(current_token).unwrap();
        generated_ids.push(current_token);
        
        if current_token == adapter.encoder.as_ref().unwrap().eos_token_id() {
            break;
        }
    }
    
    let output = adapter.decode(&generated_ids).unwrap();
    
    // 4. Validate output
    assert!(!output.is_empty());
    
    println!("Phi-3 output: {}", output);
    
    // 5. Unload
    adapter.unload().unwrap();
}

#[test]
fn gate3_adapter_registry_integration() {
    // 1. Register Llama adapter
    let registry = AdapterRegistry::new();
    registry.register("llama/qwen", || Box::new(LlamaInferenceAdapter::new(LlamaVariant::Qwen)));
    registry.register("llama/phi3", || Box::new(LlamaInferenceAdapter::new(LlamaVariant::Phi3)));
    
    // 2. Create adapter via registry
    let mut qwen_adapter = registry.create("llama/qwen").unwrap();
    qwen_adapter.load(Path::new("qwen2.5-0.5b.gguf")).unwrap();
    
    let input_ids = qwen_adapter.encode("Hello").unwrap();
    let output = qwen_adapter.prefill(&input_ids).unwrap();
    
    assert!(!output.is_empty());
    
    qwen_adapter.unload().unwrap();
}

#[test]
fn gate3_adapter_polymorphism() {
    // Test using adapter as trait object
    let adapters: Vec<Box<dyn InferenceAdapter>> = vec![
        Box::new(LlamaInferenceAdapter::new(LlamaVariant::Qwen)),
        Box::new(LlamaInferenceAdapter::new(LlamaVariant::Phi3)),
    ];
    
    for adapter in adapters {
        assert!(!adapter.model_type().is_empty());
    }
}
```

---

## Testing Strategy

### Validation Tests
- Run Gate 3 validation checklist
- Verify all items pass
- Document any issues

### Integration Tests
- Test Qwen via adapter
- Test Phi-3 via adapter
- Test adapter registry integration
- Test polymorphic usage

### Quality Tests
- Compare adapter output with direct implementation
- Verify no quality degradation
- Verify no performance regression

### Manual Verification
1. Run Gate 3 validation test suite
2. Verify all tests pass
3. Review Gate 3 checklist
4. Generate Gate 3 report
5. Get sign-off from Foundation-Alpha

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Gate 3 validation checklist complete
- [ ] All validation tests passing
- [ ] No blocking issues
- [ ] Gate 3 report generated
- [ ] Sign-off from Foundation-Alpha team
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 7 (Integration)
- Related Stories: LT-033, FT-028, FT-029

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

**Gate 3 Milestone**: This checkpoint validates that the adapter pattern is complete and both Llama models work through unified interface. This enables final integration and testing.

---

Detailed by Project Management Team — ready to implement 📋
