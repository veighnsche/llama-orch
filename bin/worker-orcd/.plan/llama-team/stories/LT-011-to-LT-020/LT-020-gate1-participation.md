# LT-020: Gate 1 Participation

**Team**: Llama-Beta  
**Sprint**: Sprint 4 - GQA Attention + Integration  
**Size**: S (1 day)  
**Days**: 54  
**Spec Ref**: Gate 1

---

## Story Description

Participate in Gate 1 validation to verify Llama kernels are complete and integrated with Foundation-Alpha's infrastructure. Validate all Llama-specific kernels work correctly with FFI layer, KV cache, and integration test framework.

---

## Acceptance Criteria

- [ ] All Llama kernels integrated with FFI layer
- [ ] RoPE kernel callable from Rust via FFI
- [ ] RMSNorm kernel callable from Rust via FFI
- [ ] Residual kernel callable from Rust via FFI
- [ ] GQA Attention kernels callable from Rust via FFI
- [ ] SwiGLU FFN kernel callable from Rust via FFI
- [ ] All kernels work with Foundation's KV cache
- [ ] Integration tests pass (FT-024: HTTP-FFI-CUDA)
- [ ] Gate 1 validation checklist complete
- [ ] No blocking issues for Qwen integration
- [ ] Documentation updated with integration status
- [ ] Sign-off from Foundation-Alpha team

---

## Dependencies

### Upstream (Blocks This Story)
- LT-012: RoPE Kernel (needs kernel)
- LT-013: RMSNorm Kernel (needs kernel)
- LT-014: Residual Kernel (needs kernel)
- LT-015: GQA Attention Prefill (needs kernel)
- LT-016: GQA Attention Decode (needs kernel)
- LT-017: SwiGLU FFN (needs kernel)
- LT-019: Kernel Unit Tests (needs validated kernels)
- FT-024: HTTP-FFI-CUDA Integration Test (needs integration framework)
- FT-027: Gate 1 Checkpoint (needs gate validation)

### Downstream (This Story Blocks)
- LT-022: Qwen Weight Mapping (needs validated kernels)
- LT-024: Qwen Forward Pass (needs integrated kernels)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/.plan/llama-team/integration-gates/gate-1-llama-kernels.md` - Gate 1 report
- `bin/worker-orcd/src/kernels/mod.rs` - Kernel module exports
- `bin/worker-orcd/tests/integration/gate1_llama.rs` - Gate 1 integration tests

### Gate 1 Validation Checklist

**FFI Integration**:
- [ ] RoPE kernel FFI binding works
- [ ] RMSNorm kernel FFI binding works
- [ ] Residual kernel FFI binding works
- [ ] GQA Prefill kernel FFI binding works
- [ ] GQA Decode kernel FFI binding works
- [ ] SwiGLU kernel FFI binding works
- [ ] All kernels handle errors correctly (Rust Result<T, E>)
- [ ] All kernels log via tracing (no printf)

**KV Cache Integration**:
- [ ] GQA Prefill writes to KV cache correctly
- [ ] GQA Decode reads from KV cache correctly
- [ ] KV cache allocation works for Llama models
- [ ] KV cache management works across prefill/decode

**Integration Tests**:
- [ ] HTTP → Rust → FFI → CUDA → Rust → HTTP works
- [ ] Error propagation works (CUDA → Rust → HTTP)
- [ ] Correlation ID tracked through all layers
- [ ] Memory cleanup works (no leaks)

**Performance**:
- [ ] All kernels meet performance targets
- [ ] No unexpected latency spikes
- [ ] VRAM usage within limits

**Documentation**:
- [ ] All kernels documented (API, usage, examples)
- [ ] Integration guide updated
- [ ] Known issues documented

### Integration Test
```rust
#[test]
fn gate1_llama_kernels_integration() {
    // Setup
    let cuda_ctx = CudaContext::new().unwrap();
    let kv_cache = KVCache::new(1, 32768, 2, 64).unwrap();
    
    // Test RoPE
    let q = allocate_device(10 * 14 * 64);
    let config = RoPEConfig { seq_len: 10, num_heads: 14, head_dim: 64, freq_base: 10000.0, rope_dim: 64 };
    rope_forward(q, q, &config).unwrap();
    
    // Test RMSNorm
    let x = allocate_device(10 * 896);
    let weight = allocate_device(896);
    let config = RMSNormConfig { batch_size: 1, seq_len: 10, hidden_dim: 896, eps: 1e-6 };
    rmsnorm_forward(x, x, weight, &config).unwrap();
    
    // Test GQA Attention
    let q = allocate_device(1 * 10 * 14 * 64);
    let k = allocate_device(1 * 10 * 2 * 64);
    let v = allocate_device(1 * 10 * 2 * 64);
    let output = allocate_device(1 * 10 * 14 * 64);
    let config = GQAAttentionConfig { batch_size: 1, seq_len: 10, num_q_heads: 14, num_kv_heads: 2, head_dim: 64, scale: 0.125 };
    gqa_attention_prefill(output, q, k, v, kv_cache.k_ptr(), kv_cache.v_ptr(), &config).unwrap();
    
    // Verify all operations succeeded
    assert!(true);
}
```

---

## Testing Strategy

### Integration Tests
- Test all kernels via FFI
- Test KV cache integration
- Test error handling
- Test HTTP → CUDA → HTTP flow

### Validation Tests
- Run Gate 1 validation checklist
- Verify all items pass
- Document any issues

### Manual Verification
1. Run Gate 1 integration test suite
2. Verify all tests pass
3. Review Gate 1 checklist
4. Get sign-off from Foundation-Alpha

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Gate 1 validation checklist complete
- [ ] All integration tests passing
- [ ] No blocking issues
- [ ] Documentation updated
- [ ] Sign-off from Foundation-Alpha team
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 7 (Integration)
- Gate 1 Checklist: `bin/worker-orcd/.plan/foundation-team/integration-gates/gate-1-foundation-complete.md`
- Related Stories: LT-012 through LT-019, FT-024, FT-027

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

**Gate 1 Milestone**: This story marks completion of Llama kernel development. All kernels must be validated and integrated before proceeding to Qwen model implementation.

---

Detailed by Project Management Team — ready to implement 📋
