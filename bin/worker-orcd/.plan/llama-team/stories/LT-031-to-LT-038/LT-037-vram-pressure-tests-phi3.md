# LT-037: VRAM Pressure Tests (Phi-3)

**Team**: Llama-Beta  
**Sprint**: Sprint 7 - Final Integration  
**Size**: M (2 days)  
**Days**: 83-84  
**Spec Ref**: M0-W-1230

---

## Story Description

Test Phi-3 model under VRAM pressure conditions to validate memory management, KV cache handling, and graceful degradation. Ensure worker handles large context lengths and memory constraints correctly.

---

## Acceptance Criteria

- [ ] Test Phi-3 with maximum context length (4096 tokens)
- [ ] Test Phi-3 with near-maximum VRAM usage
- [ ] Test KV cache growth to maximum size
- [ ] Test memory allocation failures (graceful handling)
- [ ] Test context length overflow (reject gracefully)
- [ ] Measure VRAM usage at various context lengths
- [ ] Validate no memory leaks during long generation
- [ ] Unit tests validate VRAM management
- [ ] Integration tests validate pressure scenarios
- [ ] Error handling for OOM conditions
- [ ] Log VRAM usage and pressure events

---

## Dependencies

### Upstream (Blocks This Story)
- LT-036: Reproducibility Tests (needs stable baseline)
- LT-031: Phi-3 Forward Pass (needs Phi-3 model)

### Downstream (This Story Blocks)
- LT-038: Documentation (needs test results)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/tests/integration/vram_pressure_phi3.rs` - VRAM pressure tests
- `bin/worker-orcd/.docs/vram_pressure_report.md` - Pressure test report

### Test Scenarios

**1. Maximum Context Length Test**:
```rust
#[test]
fn test_phi3_max_context_length() {
    let mut adapter = LlamaInferenceAdapter::from_model_ref("phi-3-mini-4k").unwrap();
    adapter.load(Path::new("phi-3-mini-4k.gguf")).unwrap();
    
    // Generate 4096-token prompt (max context)
    let long_prompt = generate_long_prompt(4000);
    let input_ids = adapter.encode(&long_prompt).unwrap();
    
    tracing::info!("Testing with {} tokens (near max context)", input_ids.len());
    
    // Should succeed
    let result = adapter.prefill(&input_ids);
    assert!(result.is_ok());
    
    // Try to generate more tokens (should hit limit)
    let mut current_token = *result.unwrap().last().unwrap();
    
    for i in 0..100 {
        let result = adapter.decode_token(current_token);
        
        if input_ids.len() + i >= 4096 {
            // Should fail gracefully at context limit
            assert!(result.is_err());
            break;
        }
        
        current_token = result.unwrap();
    }
    
    adapter.unload().unwrap();
}
```

**2. KV Cache Growth Test**:
```rust
#[test]
fn test_phi3_kv_cache_growth() {
    let mut adapter = LlamaInferenceAdapter::from_model_ref("phi-3-mini-4k").unwrap();
    adapter.load(Path::new("phi-3-mini-4k.gguf")).unwrap();
    
    let initial_vram = get_vram_usage();
    
    // Generate tokens and monitor KV cache growth
    let input_ids = adapter.encode("Hello").unwrap();
    adapter.prefill(&input_ids).unwrap();
    
    let mut vram_measurements = Vec::new();
    
    for i in 0..1000 {
        let current_vram = get_vram_usage();
        vram_measurements.push(current_vram);
        
        adapter.decode_token(0).unwrap();
        
        if i % 100 == 0 {
            tracing::info!("Position {}: VRAM usage = {} MB", i, current_vram / (1024 * 1024));
        }
    }
    
    // Verify linear growth (KV cache)
    let growth_rate = (vram_measurements.last().unwrap() - vram_measurements.first().unwrap()) / 1000;
    tracing::info!("KV cache growth rate: {} bytes/token", growth_rate);
    
    // Expected: ~12 KB/token for Phi-3 (32 heads × 96 dim × 2 bytes × 2 (K+V))
    assert!(growth_rate > 10_000 && growth_rate < 15_000);
    
    adapter.unload().unwrap();
}
```

**3. VRAM Allocation Failure Test**:
```rust
#[test]
fn test_phi3_vram_allocation_failure() {
    // Simulate low VRAM condition
    let available_vram = 1_000_000_000; // 1 GB (insufficient for Phi-3)
    
    let result = Phi3Loader::load_with_vram_limit("phi-3-mini-4k.gguf", available_vram);
    
    // Should fail gracefully
    assert!(result.is_err());
    
    match result {
        Err(LoadError::InsufficientVRAM { required, available }) => {
            tracing::info!("Correctly rejected: required={} MB, available={} MB", 
                          required / (1024 * 1024), available / (1024 * 1024));
            assert!(required > available);
        }
        _ => panic!("Expected InsufficientVRAM error"),
    }
}
```

**4. Context Overflow Test**:
```rust
#[test]
fn test_phi3_context_overflow() {
    let mut adapter = LlamaInferenceAdapter::from_model_ref("phi-3-mini-4k").unwrap();
    adapter.load(Path::new("phi-3-mini-4k.gguf")).unwrap();
    
    // Try to encode prompt longer than context
    let very_long_prompt = generate_long_prompt(5000); // > 4096
    let input_ids = adapter.encode(&very_long_prompt).unwrap();
    
    tracing::info!("Encoded {} tokens (exceeds context limit)", input_ids.len());
    
    // Should fail gracefully
    let result = adapter.prefill(&input_ids);
    
    assert!(result.is_err());
    match result {
        Err(AdapterError::ContextLengthExceeded { length, max }) => {
            tracing::info!("Correctly rejected: length={}, max={}", length, max);
            assert_eq!(max, 4096);
        }
        _ => panic!("Expected ContextLengthExceeded error"),
    }
    
    adapter.unload().unwrap();
}
```

**5. Memory Leak Test**:
```rust
#[test]
fn test_phi3_no_memory_leaks() {
    let initial_vram = get_vram_usage();
    
    // Load and unload 10 times
    for i in 0..10 {
        let mut adapter = LlamaInferenceAdapter::from_model_ref("phi-3-mini-4k").unwrap();
        adapter.load(Path::new("phi-3-mini-4k.gguf")).unwrap();
        
        // Generate some tokens
        let input_ids = adapter.encode("Hello").unwrap();
        adapter.prefill(&input_ids).unwrap();
        
        for _ in 0..10 {
            adapter.decode_token(0).unwrap();
        }
        
        adapter.unload().unwrap();
        
        let current_vram = get_vram_usage();
        tracing::info!("Iteration {}: VRAM usage = {} MB", i, current_vram / (1024 * 1024));
    }
    
    let final_vram = get_vram_usage();
    let leak = final_vram.saturating_sub(initial_vram);
    
    tracing::info!("VRAM leak: {} MB", leak / (1024 * 1024));
    
    // Allow small leak (<100 MB) due to CUDA allocator
    assert!(leak < 100_000_000, "Memory leak detected: {} MB", leak / (1024 * 1024));
}
```

**6. Concurrent Context Test**:
```rust
#[test]
fn test_phi3_concurrent_contexts() {
    // Test multiple contexts (batch_size > 1)
    let mut adapter = LlamaInferenceAdapter::from_model_ref("phi-3-mini-4k").unwrap();
    adapter.load(Path::new("phi-3-mini-4k.gguf")).unwrap();
    
    // Note: Current implementation is batch_size=1
    // This test validates that batch_size > 1 is rejected or handled correctly
    
    let config = ForwardPassConfig {
        is_prefill: true,
        batch_size: 4, // Try batch processing
        seq_len: 10,
        cache_len: 0,
        temperature: 0.7,
        seed: 42,
    };
    
    let input_ids = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let result = adapter.prefill(&input_ids);
    
    // Should either work or fail gracefully
    if result.is_err() {
        tracing::info!("Batch processing not supported (expected)");
    } else {
        tracing::info!("Batch processing supported");
    }
    
    adapter.unload().unwrap();
}
```

### VRAM Usage Measurements

**Expected VRAM Usage (Phi-3-mini-4k)**:
- Model weights: ~7.5 GB
- KV cache (empty): ~0 MB
- KV cache (1000 tokens): ~12 MB
- KV cache (4096 tokens): ~49 MB
- Total (max context): ~7.6 GB

**Measurement Points**:
```rust
fn get_vram_usage() -> usize {
    // Query CUDA for current VRAM usage
    let mut free: usize = 0;
    let mut total: usize = 0;
    
    unsafe {
        cudaMemGetInfo(&mut free, &mut total);
    }
    
    total - free
}
```

---

## Testing Strategy

### Pressure Tests
- Test maximum context length (4096 tokens)
- Test KV cache growth (0 → 4096 tokens)
- Test VRAM allocation failures
- Test context overflow
- Test memory leaks (10 load/unload cycles)

### Validation Tests
- Verify graceful error handling
- Verify no crashes under pressure
- Verify VRAM cleanup on unload

### Performance Tests
- Measure VRAM usage at various context lengths
- Measure KV cache growth rate
- Measure memory leak rate

### Manual Verification
1. Run VRAM pressure test suite
2. Monitor VRAM usage with nvidia-smi
3. Verify no OOM crashes
4. Review pressure test report

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] All pressure tests passing (6+ tests)
- [ ] VRAM pressure report generated
- [ ] No memory leaks detected
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.6 (Model Loading)
- Related Stories: LT-036, LT-031

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋
