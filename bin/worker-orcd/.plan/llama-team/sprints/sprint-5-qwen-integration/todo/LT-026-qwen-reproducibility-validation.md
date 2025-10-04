# LT-026: Qwen Reproducibility Validation

**Team**: Llama-Beta  
**Sprint**: Sprint 5 - Qwen Integration  
**Size**: M (2 days)  
**Days**: 66-67  
**Spec Ref**: M0-W-1826

---

## Story Description

Validate Qwen2.5-0.5B model reproducibility by running identical prompts with fixed seeds and verifying deterministic output. Ensure seeded RNG produces identical token sequences across multiple runs.

---

## Acceptance Criteria

- [ ] Run same prompt 10 times with fixed seed (seed=42)
- [ ] Verify all 10 runs produce identical token sequences
- [ ] Test with 3 different prompts
- [ ] Test with 3 different seeds (42, 123, 999)
- [ ] Validate byte-for-byte output equality
- [ ] Record token sequences for each run
- [ ] Create reproducibility report with pass/fail
- [ ] Unit tests validate seeded RNG
- [ ] Integration tests validate end-to-end reproducibility
- [ ] Error handling for non-deterministic behavior
- [ ] Log reproducibility test results

---

## Dependencies

### Upstream (Blocks This Story)
- LT-025: Qwen Haiku Generation Test (needs working generation)
- FT-020: Seeded RNG (needs deterministic sampling)

### Downstream (This Story Blocks)
- LT-027: Gate 2 Checkpoint (needs reproducibility validation)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/tests/integration/qwen_reproducibility_test.cpp` - Reproducibility tests
- `bin/worker-orcd/tests/integration/qwen_reproducibility_test.rs` - Rust test wrapper
- `bin/worker-orcd/.docs/qwen_reproducibility_report.md` - Test report

### Test Structure
```cpp
struct ReproducibilityTest {
    std::string prompt;
    uint32_t seed;
    int num_runs;
    int max_tokens;
    float temperature;
};

const std::vector<ReproducibilityTest> REPRO_TESTS = {
    {
        "Write a haiku about autumn leaves",
        42,
        10,
        30,
        0.7
    },
    {
        "The quick brown fox",
        123,
        10,
        20,
        0.8
    },
    {
        "Once upon a time",
        999,
        10,
        25,
        0.5
    },
};
```

### Test Implementation
```cpp
void test_reproducibility() {
    auto model = QwenLoader::load("qwen2.5-0.5b.gguf");
    auto encoder = BPEEncoder::from_gguf("qwen2.5-0.5b.gguf");
    
    for (const auto& test : REPRO_TESTS) {
        tracing::info!("Testing reproducibility: prompt='{}', seed={}", test.prompt, test.seed);
        
        std::vector<std::vector<uint32_t>> all_runs;
        
        // Run N times with same seed
        for (int run = 0; run < test.num_runs; ++run) {
            // Reset KV cache for each run
            auto kv_cache = KVCache::new(1, 32768, 2, 64);
            
            // Encode prompt
            auto input_ids = encoder.encode(test.prompt);
            
            // Prefill
            ForwardPassConfig config = {
                true,  // is_prefill
                1,     // batch_size
                input_ids.size(),  // seq_len
                0,     // cache_len
                test.temperature,
                test.seed  // Fixed seed
            };
            auto prefill_output = QwenForward::prefill(model, input_ids, kv_cache, config);
            
            // Decode
            std::vector<uint32_t> generated_ids;
            uint32_t current_token = prefill_output.back();
            
            config.is_prefill = false;
            config.seq_len = 1;
            
            for (int i = 0; i < test.max_tokens; ++i) {
                config.cache_len = input_ids.size() + i;
                current_token = QwenForward::decode(model, current_token, kv_cache, config);
                generated_ids.push_back(current_token);
                
                if (current_token == encoder.eos_token_id()) {
                    break;
                }
            }
            
            all_runs.push_back(generated_ids);
            tracing::debug!("Run {}: {} tokens generated", run, generated_ids.size());
        }
        
        // Validate all runs are identical
        bool all_identical = true;
        for (int run = 1; run < test.num_runs; ++run) {
            if (all_runs[run] != all_runs[0]) {
                all_identical = false;
                tracing::error!("Run {} differs from run 0", run);
                
                // Log differences
                for (size_t i = 0; i < std::min(all_runs[0].size(), all_runs[run].size()); ++i) {
                    if (all_runs[0][i] != all_runs[run][i]) {
                        tracing::error!("  Position {}: {} != {}", i, all_runs[0][i], all_runs[run][i]);
                    }
                }
            }
        }
        
        ASSERT_TRUE(all_identical);
        tracing::info!("✅ Reproducibility validated: {} runs identical", test.num_runs);
    }
}
```

### Reproducibility Report
```markdown
# Qwen2.5-0.5B Reproducibility Report

**Date**: 2025-10-04  
**Model**: Qwen2.5-0.5B-Instruct  
**Test**: 10 runs per prompt with fixed seed

## Test Results

### Test 1: "Write a haiku about autumn leaves" (seed=42)
- **Runs**: 10
- **Result**: ✅ PASS (all identical)
- **Token sequence**: [9906, 264, 47218, 922, 42150, 11141, ...]
- **Output**: "Autumn leaves fall / Golden colors paint the ground / Nature's art displayed"

### Test 2: "The quick brown fox" (seed=123)
- **Runs**: 10
- **Result**: ✅ PASS (all identical)
- **Token sequence**: [791, 4062, 14198, 39935, 35308, ...]
- **Output**: "The quick brown fox jumps over the lazy dog..."

### Test 3: "Once upon a time" (seed=999)
- **Runs**: 10
- **Result**: ✅ PASS (all identical)
- **Token sequence**: [12805, 5304, 264, 892, 1070, ...]
- **Output**: "Once upon a time in a faraway land..."

## Summary
- **Total tests**: 3
- **Passed**: 3
- **Failed**: 0
- **Reproducibility**: 100%
```

### Seeded RNG Validation
```cpp
void test_seeded_rng() {
    // Test that seeded RNG produces same sequence
    uint32_t seed = 42;
    
    // Run 1
    auto rng1 = SeededRNG::new(seed);
    std::vector<float> samples1;
    for (int i = 0; i < 100; ++i) {
        samples1.push_back(rng1.uniform(0.0f, 1.0f));
    }
    
    // Run 2 (same seed)
    auto rng2 = SeededRNG::new(seed);
    std::vector<float> samples2;
    for (int i = 0; i < 100; ++i) {
        samples2.push_back(rng2.uniform(0.0f, 1.0f));
    }
    
    // Validate identical
    for (int i = 0; i < 100; ++i) {
        ASSERT_FLOAT_EQ(samples1[i], samples2[i]);
    }
}
```

---

## Testing Strategy

### Unit Tests
- Test seeded RNG produces identical sequences
- Test sampling with fixed seed
- Test temperature scaling with fixed seed

### Integration Tests
- Test full generation reproducibility (10 runs)
- Test with 3 different prompts
- Test with 3 different seeds
- Test byte-for-byte output equality

### Validation Tests
- Verify token sequences are identical
- Verify decoded text is identical
- Verify no floating-point drift
- Verify KV cache doesn't affect reproducibility

### Manual Verification
1. Run reproducibility test suite
2. Verify all tests pass
3. Review reproducibility report
4. Check logs for any warnings

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] Unit tests passing (3+ tests)
- [ ] Integration tests passing
- [ ] All reproducibility tests pass (100%)
- [ ] Reproducibility report generated
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.9 (Reproducibility)
- Related Stories: LT-025, FT-020

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋
