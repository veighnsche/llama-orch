# LT-035: Llama Integration Test Suite

**Team**: Llama-Beta  
**Sprint**: Sprint 7 - Final Integration  
**Size**: M (3 days)  
**Days**: 78-80  
**Spec Ref**: M0-W-1818

---

## Story Description

Create comprehensive integration test suite for Llama team deliverables. Test end-to-end workflows including GGUF loading, tokenization, inference, and adapter usage for both Qwen and Phi-3 models.

---

## Acceptance Criteria

- [ ] Create integration test suite for GGUF loading
- [ ] Create integration test suite for tokenization (BPE)
- [ ] Create integration test suite for Llama kernels
- [ ] Create integration test suite for Qwen model
- [ ] Create integration test suite for Phi-3 model
- [ ] Create integration test suite for LlamaInferenceAdapter
- [ ] Test HTTP → Rust → FFI → CUDA → Rust → HTTP flow
- [ ] Test error handling and recovery
- [ ] All integration tests pass
- [ ] Test coverage report generated
- [ ] Documentation updated with test guide
- [ ] Log test results with pass/fail status

---

## Dependencies

### Upstream (Blocks This Story)
- LT-034: Gate 3 Participation (needs validated adapter)
- LT-026: Qwen Reproducibility Validation (needs Qwen tests)
- LT-032: Tokenizer Conformance Tests Phi-3 (needs Phi-3 tests)

### Downstream (This Story Blocks)
- LT-036: Reproducibility Tests (needs test infrastructure)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/tests/integration/gguf_loading_tests.rs` - GGUF loading tests
- `bin/worker-orcd/tests/integration/tokenization_tests.rs` - Tokenization tests
- `bin/worker-orcd/tests/integration/kernel_tests.rs` - Kernel integration tests
- `bin/worker-orcd/tests/integration/qwen_e2e_tests.rs` - Qwen end-to-end tests
- `bin/worker-orcd/tests/integration/phi3_e2e_tests.rs` - Phi-3 end-to-end tests
- `bin/worker-orcd/tests/integration/adapter_tests.rs` - Adapter integration tests
- `bin/worker-orcd/.docs/llama_integration_test_guide.md` - Test documentation

### Test Categories

**1. GGUF Loading Tests** (5 tests):
```rust
#[test]
fn test_gguf_header_parsing() {
    let mmap = MmapFile::open("qwen2.5-0.5b.gguf").unwrap();
    let header = parse_gguf_header(&mmap).unwrap();
    
    assert_eq!(header.magic, 0x47475546);
    assert_eq!(header.version, 3);
    assert!(header.tensor_count > 0);
}

#[test]
fn test_gguf_metadata_extraction() {
    let mmap = MmapFile::open("qwen2.5-0.5b.gguf").unwrap();
    let metadata = parse_gguf_metadata(&mmap).unwrap();
    
    assert_eq!(metadata.get_string("general.architecture"), "llama");
    assert_eq!(metadata.get_uint32("llama.block_count"), 24);
}

#[test]
fn test_gguf_weight_mapping() {
    let weights = map_qwen_weights("qwen2.5-0.5b.gguf").unwrap();
    
    assert!(weights.token_embedding.is_some());
    assert_eq!(weights.layers.len(), 24);
}

#[test]
fn test_gguf_weight_loading() {
    let model = QwenLoader::load("qwen2.5-0.5b.gguf").unwrap();
    
    assert!(model.total_vram_bytes > 800_000_000);
    assert!(model.total_vram_bytes < 1_000_000_000);
}

#[test]
fn test_gguf_security_validation() {
    // Test with malicious GGUF (invalid offsets)
    let result = QwenLoader::load("malicious.gguf");
    
    assert!(result.is_err());
    // Should reject due to bounds validation
}
```

**2. Tokenization Tests** (6 tests):
```rust
#[test]
fn test_bpe_encoding() {
    let encoder = BPEEncoder::from_gguf("qwen2.5-0.5b.gguf").unwrap();
    let ids = encoder.encode("Hello, world!");
    
    assert!(!ids.is_empty());
}

#[test]
fn test_bpe_decoding() {
    let decoder = BPEDecoder::from_gguf("qwen2.5-0.5b.gguf").unwrap();
    let text = decoder.decode(&[9906, 11, 1917, 0]).unwrap();
    
    assert!(!text.is_empty());
}

#[test]
fn test_bpe_round_trip() {
    let encoder = BPEEncoder::from_gguf("qwen2.5-0.5b.gguf").unwrap();
    let decoder = BPEDecoder::from_gguf("qwen2.5-0.5b.gguf").unwrap();
    
    let original = "Hello, world!";
    let ids = encoder.encode(original);
    let decoded = decoder.decode(&ids).unwrap();
    
    assert_eq!(decoded, original);
}

#[test]
fn test_utf8_streaming_decode() {
    let decoder = BPEDecoder::from_gguf("qwen2.5-0.5b.gguf").unwrap();
    let mut streaming = StreamingDecoder::new(decoder);
    
    // Test with multi-byte UTF-8
    let token_id = 108386; // Chinese character
    let output = streaming.decode_token(token_id);
    
    // Should not have broken UTF-8
    assert!(output.is_empty() || output.chars().all(|c| c.is_valid()));
}

#[test]
fn test_tokenizer_conformance_qwen() {
    run_qwen_conformance_tests();
    // All test vectors should pass
}

#[test]
fn test_tokenizer_conformance_phi3() {
    run_phi3_conformance_tests();
    // All test vectors should pass
}
```

**3. Kernel Integration Tests** (7 tests):
```rust
#[test]
fn test_rope_kernel_integration() {
    let q = allocate_and_fill_random(10 * 14 * 64);
    let config = RoPEConfig { seq_len: 10, num_heads: 14, head_dim: 64, freq_base: 10000.0, rope_dim: 64 };
    
    rope_forward(q, q, &config).unwrap();
    
    // Verify rotation applied
}

#[test]
fn test_rmsnorm_kernel_integration() {
    let x = allocate_and_fill_random(10 * 896);
    let weight = allocate_and_fill_ones(896);
    let config = RMSNormConfig { batch_size: 1, seq_len: 10, hidden_dim: 896, eps: 1e-6 };
    
    rmsnorm_forward(x, x, weight, &config).unwrap();
    
    // Verify normalization applied
}

#[test]
fn test_gqa_attention_integration() {
    let q = allocate_and_fill_random(1 * 10 * 14 * 64);
    let k = allocate_and_fill_random(1 * 10 * 2 * 64);
    let v = allocate_and_fill_random(1 * 10 * 2 * 64);
    let kv_cache = KVCache::new(1, 32768, 2, 64).unwrap();
    let config = GQAAttentionConfig { batch_size: 1, seq_len: 10, num_q_heads: 14, num_kv_heads: 2, head_dim: 64, scale: 0.125 };
    
    let output = gqa_attention_prefill(q, k, v, &kv_cache, &config).unwrap();
    
    assert!(!output.is_empty());
}

#[test]
fn test_swiglu_ffn_integration() {
    let x = allocate_and_fill_random(10 * 896);
    let w_gate = allocate_and_fill_random(4864 * 896);
    let w_up = allocate_and_fill_random(4864 * 896);
    let w_down = allocate_and_fill_random(896 * 4864);
    let config = SwiGLUConfig { batch_size: 1, seq_len: 10, hidden_dim: 896, ffn_dim: 4864 };
    
    let output = swiglu_ffn_forward(x, w_gate, w_up, w_down, &config).unwrap();
    
    assert!(!output.is_empty());
}

// ... more kernel tests
```

**4. Qwen End-to-End Tests** (3 tests):
```rust
#[test]
fn test_qwen_haiku_generation() {
    let model = QwenLoader::load("qwen2.5-0.5b.gguf").unwrap();
    let encoder = BPEEncoder::from_gguf("qwen2.5-0.5b.gguf").unwrap();
    let decoder = BPEDecoder::from_gguf("qwen2.5-0.5b.gguf").unwrap();
    
    let prompt = "Write a haiku about autumn leaves";
    let input_ids = encoder.encode(prompt);
    
    let mut kv_cache = KVCache::new(1, 32768, 2, 64).unwrap();
    let generated_ids = generate_tokens(&model, &input_ids, &mut kv_cache, 30);
    
    let output = decoder.decode(&generated_ids).unwrap();
    
    assert!(!output.is_empty());
    assert!(output.len() > 10);
}

#[test]
fn test_qwen_reproducibility() {
    // Run same prompt 3 times with same seed
    let outputs = (0..3).map(|_| {
        generate_with_seed("qwen2.5-0.5b.gguf", "Hello", 42, 20)
    }).collect::<Vec<_>>();
    
    assert_eq!(outputs[0], outputs[1]);
    assert_eq!(outputs[1], outputs[2]);
}

#[test]
fn test_qwen_http_integration() {
    // Test HTTP → Rust → FFI → CUDA → Rust → HTTP
    let server = start_test_server();
    
    let response = client.post("/execute")
        .json(&json!({
            "model_ref": "qwen2.5-0.5b-instruct",
            "prompt": "Hello, world!",
            "max_tokens": 10
        }))
        .send()
        .unwrap();
    
    assert_eq!(response.status(), 200);
    let body: serde_json::Value = response.json().unwrap();
    assert!(body["output"].as_str().unwrap().len() > 0);
}
```

**5. Phi-3 End-to-End Tests** (3 tests):
- Similar to Qwen tests but for Phi-3

**6. Adapter Integration Tests** (4 tests):
```rust
#[test]
fn test_adapter_qwen_generation() {
    let mut adapter = LlamaInferenceAdapter::from_model_ref("qwen2.5-0.5b").unwrap();
    adapter.load(Path::new("qwen2.5-0.5b.gguf")).unwrap();
    
    let input_ids = adapter.encode("Hello").unwrap();
    let output_ids = adapter.prefill(&input_ids).unwrap();
    
    assert!(!output_ids.is_empty());
    
    adapter.unload().unwrap();
}

#[test]
fn test_adapter_phi3_generation() {
    let mut adapter = LlamaInferenceAdapter::from_model_ref("phi-3-mini-4k").unwrap();
    adapter.load(Path::new("phi-3-mini-4k.gguf")).unwrap();
    
    let input_ids = adapter.encode("Hello").unwrap();
    let output_ids = adapter.prefill(&input_ids).unwrap();
    
    assert!(!output_ids.is_empty());
    
    adapter.unload().unwrap();
}

#[test]
fn test_adapter_model_switching() {
    // Load Qwen
    let mut adapter = LlamaInferenceAdapter::from_model_ref("qwen2.5-0.5b").unwrap();
    adapter.load(Path::new("qwen2.5-0.5b.gguf")).unwrap();
    adapter.unload().unwrap();
    
    // Switch to Phi-3
    adapter = LlamaInferenceAdapter::from_model_ref("phi-3-mini-4k").unwrap();
    adapter.load(Path::new("phi-3-mini-4k.gguf")).unwrap();
    adapter.unload().unwrap();
}

#[test]
fn test_adapter_registry() {
    let registry = AdapterRegistry::new();
    registry.register_llama_adapters();
    
    let qwen = registry.create("llama/qwen").unwrap();
    let phi3 = registry.create("llama/phi3").unwrap();
    
    assert_eq!(qwen.model_type(), "llama/qwen");
    assert_eq!(phi3.model_type(), "llama/phi3");
}
```

---

## Testing Strategy

### Integration Tests
- Run all 28+ integration tests
- Verify all tests pass
- Generate test coverage report

### Performance Tests
- Measure test execution time
- Ensure tests complete in reasonable time (<5 min total)

### Manual Verification
1. Run full integration test suite
2. Verify all tests pass
3. Review test coverage report
4. Check logs for any warnings

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] All integration tests passing (28+ tests)
- [ ] Test coverage report generated
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.11 (Testing)
- Related Stories: LT-034, LT-026, LT-032

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋
