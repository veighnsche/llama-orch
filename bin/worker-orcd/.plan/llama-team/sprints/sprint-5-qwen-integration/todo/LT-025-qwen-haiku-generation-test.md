# LT-025: Qwen Haiku Generation Test

**Team**: Llama-Beta  
**Sprint**: Sprint 5 - Qwen Integration  
**Size**: M (2 days)  
**Days**: 64-65  
**Spec Ref**: M0-W-1800

---

## Story Description

Create end-to-end haiku generation test for Qwen2.5-0.5B model. Validate complete pipeline from prompt encoding through token generation to decoded output, ensuring model produces coherent haiku-style text.

---

## Acceptance Criteria

- [ ] Create haiku generation test with prompt "Write a haiku about"
- [ ] Encode prompt using BPE tokenizer
- [ ] Run prefill forward pass with encoded prompt
- [ ] Generate 20-30 tokens autoregressively (decode)
- [ ] Decode generated tokens to UTF-8 text
- [ ] Validate output is coherent haiku-style text
- [ ] Test with multiple haiku prompts (3-5 variations)
- [ ] Measure generation latency (tokens per second)
- [ ] Validate streaming decode (UTF-8 safe)
- [ ] Unit tests validate generation pipeline
- [ ] Integration tests validate end-to-end flow
- [ ] Error handling for generation failures
- [ ] Log generation results and timing

---

## Dependencies

### Upstream (Blocks This Story)
- LT-024: Qwen Forward Pass (needs forward pass)
- LT-009: Byte-Level BPE Encoder (needs encoding)
- LT-010: Byte-Level BPE Decoder (needs decoding)
- LT-011: UTF-8 Safe Streaming Decode (needs streaming)
- LT-018: Tokenizer Conformance Tests (needs validated tokenizer)

### Downstream (This Story Blocks)
- LT-026: Qwen Reproducibility Validation (needs working generation)
- LT-027: Gate 2 Checkpoint (needs validated model)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/tests/integration/qwen_haiku_test.cpp` - Haiku generation test
- `bin/worker-orcd/tests/integration/qwen_haiku_test.rs` - Rust test wrapper
- `bin/worker-orcd/.docs/qwen_haiku_examples.md` - Example outputs

### Test Structure
```cpp
struct HaikuTest {
    std::string prompt;
    int max_tokens;
    float temperature;
    std::string expected_pattern;  // Regex or keywords
};

const std::vector<HaikuTest> HAIKU_TESTS = {
    {
        "Write a haiku about autumn leaves",
        30,
        0.7,
        ".*autumn.*leaves.*"  // Should mention autumn and leaves
    },
    {
        "Write a haiku about the ocean",
        30,
        0.7,
        ".*ocean.*waves.*"  // Should mention ocean/waves
    },
    {
        "Write a haiku about cherry blossoms",
        30,
        0.7,
        ".*cherry.*blossom.*"  // Should mention cherry blossoms
    },
};
```

### Test Implementation
```cpp
void test_haiku_generation() {
    // 1. Load model
    auto model = QwenLoader::load("qwen2.5-0.5b.gguf");
    auto kv_cache = KVCache::new(1, 32768, 2, 64);
    
    // 2. Load tokenizer
    auto encoder = BPEEncoder::from_gguf("qwen2.5-0.5b.gguf");
    auto decoder = BPEDecoder::from_gguf("qwen2.5-0.5b.gguf");
    auto streaming_decoder = StreamingDecoder::new(decoder);
    
    for (const auto& test : HAIKU_TESTS) {
        tracing::info!("Testing haiku: {}", test.prompt);
        
        // 3. Encode prompt
        auto input_ids = encoder.encode(test.prompt);
        tracing::info!("Encoded to {} tokens", input_ids.size());
        
        // 4. Prefill
        ForwardPassConfig config = {true, 1, input_ids.size(), 0, test.temperature, 42};
        auto prefill_output = QwenForward::prefill(model, input_ids, kv_cache, config);
        
        // 5. Decode (generate tokens)
        std::vector<uint32_t> generated_ids;
        uint32_t current_token = prefill_output.back();
        
        config.is_prefill = false;
        config.seq_len = 1;
        
        for (int i = 0; i < test.max_tokens; ++i) {
            config.cache_len = input_ids.size() + i;
            current_token = QwenForward::decode(model, current_token, kv_cache, config);
            generated_ids.push_back(current_token);
            
            // Stream decode (UTF-8 safe)
            std::string partial = streaming_decoder.decode_token(current_token);
            if (!partial.empty()) {
                std::cout << partial << std::flush;
            }
            
            // Stop on EOS
            if (current_token == encoder.eos_token_id()) {
                break;
            }
        }
        
        // 6. Final decode
        std::string remaining = streaming_decoder.flush();
        std::cout << remaining << std::endl;
        
        // 7. Validate output
        std::string full_output = decoder.decode(generated_ids);
        tracing::info!("Generated: {}", full_output);
        
        // Check pattern match
        std::regex pattern(test.expected_pattern);
        ASSERT_TRUE(std::regex_search(full_output, pattern));
        
        // 8. Measure performance
        // ... (tokens per second calculation)
    }
}
```

### Example Haiku Prompts
```
1. "Write a haiku about autumn leaves"
   Expected: 3 lines, 5-7-5 syllables, autumn/leaves theme
   
2. "Write a haiku about the ocean"
   Expected: 3 lines, ocean/waves imagery
   
3. "Write a haiku about cherry blossoms"
   Expected: 3 lines, spring/cherry blossom theme
   
4. "Write a haiku about a mountain"
   Expected: 3 lines, mountain/nature imagery
   
5. "Write a haiku about winter snow"
   Expected: 3 lines, winter/snow theme
```

### Performance Metrics
```cpp
struct GenerationMetrics {
    int prompt_tokens;
    int generated_tokens;
    float prefill_time_ms;
    float decode_time_ms;
    float tokens_per_second;
    float time_to_first_token_ms;
};

GenerationMetrics measure_generation(/* ... */) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // Prefill
    auto prefill_start = start;
    auto prefill_output = QwenForward::prefill(/* ... */);
    auto prefill_end = std::chrono::high_resolution_clock::now();
    
    // Decode
    auto decode_start = prefill_end;
    // ... generate tokens
    auto decode_end = std::chrono::high_resolution_clock::now();
    
    GenerationMetrics metrics;
    metrics.prefill_time_ms = duration_ms(prefill_start, prefill_end);
    metrics.decode_time_ms = duration_ms(decode_start, decode_end);
    metrics.tokens_per_second = generated_tokens / (metrics.decode_time_ms / 1000.0);
    metrics.time_to_first_token_ms = metrics.prefill_time_ms;
    
    return metrics;
}
```

---

## Testing Strategy

### Unit Tests
- Test prompt encoding
- Test prefill execution
- Test decode execution
- Test streaming decode
- Test output validation

### Integration Tests
- Test full haiku generation pipeline
- Test with 5 different prompts
- Test streaming output (UTF-8 safe)
- Test performance metrics

### Quality Validation
- Verify output is coherent text
- Verify haiku-like structure (3 lines)
- Verify thematic relevance to prompt
- Verify no broken UTF-8 characters

### Performance Tests
- Measure tokens per second (target: >10 TPS)
- Measure time to first token (target: <500ms)
- Measure total generation time

### Manual Verification
1. Run haiku generation test
2. Read generated haikus
3. Verify quality and coherence
4. Check performance metrics
5. Verify streaming works (no broken chars)

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing (5+ tests)
- [ ] Integration tests passing
- [ ] All 5 haiku prompts generate coherent output
- [ ] Performance metrics recorded
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.8 (End-to-End Testing)
- Related Stories: LT-024, LT-009, LT-010, LT-011, LT-018

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋
