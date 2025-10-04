# LT-036: Reproducibility Tests (10 runs × 2 models)

**Team**: Llama-Beta  
**Sprint**: Sprint 7 - Final Integration  
**Size**: M (2 days)  
**Days**: 81-82  
**Spec Ref**: M0-W-1826

---

## Story Description

Run comprehensive reproducibility validation for both Qwen and Phi-3 models. Execute 10 runs per prompt with fixed seeds for both models to ensure deterministic generation across the entire Llama pipeline.

---

## Acceptance Criteria

- [ ] Run 10 reproducibility tests for Qwen (10 runs each)
- [ ] Run 10 reproducibility tests for Phi-3 (10 runs each)
- [ ] Test with 5 different prompts per model
- [ ] Test with 3 different seeds per prompt
- [ ] Verify byte-for-byte reproducibility (100%)
- [ ] Generate reproducibility report for both models
- [ ] Document any non-deterministic behavior
- [ ] All reproducibility tests pass
- [ ] Error handling for reproducibility failures
- [ ] Log test results with pass/fail status

---

## Dependencies

### Upstream (Blocks This Story)
- LT-035: Llama Integration Test Suite (needs test infrastructure)
- LT-026: Qwen Reproducibility Validation (needs Qwen tests)
- FT-020: Seeded RNG (needs deterministic sampling)

### Downstream (This Story Blocks)
- LT-037: VRAM Pressure Tests (needs stable baseline)

---

## Technical Details

### Files to Create/Modify
- `bin/worker-orcd/tests/integration/reproducibility_qwen.rs` - Qwen reproducibility tests
- `bin/worker-orcd/tests/integration/reproducibility_phi3.rs` - Phi-3 reproducibility tests
- `bin/worker-orcd/.docs/reproducibility_report.md` - Reproducibility report

### Test Configuration
```rust
struct ReproducibilityTest {
    model: String,
    prompt: String,
    seed: u32,
    num_runs: usize,
    max_tokens: usize,
    temperature: f32,
}

const QWEN_REPRO_TESTS: &[ReproducibilityTest] = &[
    ReproducibilityTest {
        model: "qwen2.5-0.5b.gguf",
        prompt: "Write a haiku about autumn leaves",
        seed: 42,
        num_runs: 10,
        max_tokens: 30,
        temperature: 0.7,
    },
    ReproducibilityTest {
        model: "qwen2.5-0.5b.gguf",
        prompt: "The quick brown fox",
        seed: 123,
        num_runs: 10,
        max_tokens: 20,
        temperature: 0.8,
    },
    ReproducibilityTest {
        model: "qwen2.5-0.5b.gguf",
        prompt: "Once upon a time",
        seed: 999,
        num_runs: 10,
        max_tokens: 25,
        temperature: 0.5,
    },
    ReproducibilityTest {
        model: "qwen2.5-0.5b.gguf",
        prompt: "In a galaxy far, far away",
        seed: 42,
        num_runs: 10,
        max_tokens: 30,
        temperature: 0.9,
    },
    ReproducibilityTest {
        model: "qwen2.5-0.5b.gguf",
        prompt: "The meaning of life is",
        seed: 123,
        num_runs: 10,
        max_tokens: 20,
        temperature: 0.6,
    },
];

const PHI3_REPRO_TESTS: &[ReproducibilityTest] = &[
    // Similar structure for Phi-3
    ReproducibilityTest {
        model: "phi-3-mini-4k.gguf",
        prompt: "Write a haiku about ocean waves",
        seed: 42,
        num_runs: 10,
        max_tokens: 30,
        temperature: 0.7,
    },
    // ... 4 more tests
];
```

### Test Implementation
```rust
#[test]
fn test_qwen_reproducibility_suite() {
    let mut passed = 0;
    let mut failed = 0;
    
    for test in QWEN_REPRO_TESTS {
        tracing::info!("Testing Qwen reproducibility: prompt='{}', seed={}", test.prompt, test.seed);
        
        let mut all_runs = Vec::new();
        
        // Run N times with same seed
        for run in 0..test.num_runs {
            let output = generate_with_seed(
                &test.model,
                &test.prompt,
                test.seed,
                test.max_tokens,
                test.temperature,
            );
            
            all_runs.push(output);
            tracing::debug!("Run {}: {} tokens", run, all_runs[run].len());
        }
        
        // Validate all runs are identical
        let all_identical = all_runs.windows(2).all(|w| w[0] == w[1]);
        
        if all_identical {
            passed += 1;
            tracing::info!("✅ PASS: {} runs identical", test.num_runs);
        } else {
            failed += 1;
            tracing::error!("❌ FAIL: Runs differ");
            
            // Log differences
            for (i, run) in all_runs.iter().enumerate() {
                if run != &all_runs[0] {
                    tracing::error!("  Run {} differs from run 0", i);
                    log_token_differences(&all_runs[0], run);
                }
            }
        }
    }
    
    tracing::info!("Qwen reproducibility: {} passed, {} failed", passed, failed);
    assert_eq!(failed, 0, "All reproducibility tests must pass");
}

#[test]
fn test_phi3_reproducibility_suite() {
    // Similar implementation for Phi-3
    let mut passed = 0;
    let mut failed = 0;
    
    for test in PHI3_REPRO_TESTS {
        // ... (same logic as Qwen)
    }
    
    tracing::info!("Phi-3 reproducibility: {} passed, {} failed", passed, failed);
    assert_eq!(failed, 0, "All reproducibility tests must pass");
}

fn generate_with_seed(
    model_path: &str,
    prompt: &str,
    seed: u32,
    max_tokens: usize,
    temperature: f32,
) -> Vec<u32> {
    // Load model
    let mut adapter = LlamaInferenceAdapter::from_model_ref(model_path).unwrap();
    adapter.load(Path::new(model_path)).unwrap();
    
    // Encode
    let input_ids = adapter.encode(prompt).unwrap();
    
    // Prefill
    let config = ForwardPassConfig {
        is_prefill: true,
        batch_size: 1,
        seq_len: input_ids.len() as i32,
        cache_len: 0,
        temperature,
        seed,
    };
    
    let prefill_output = adapter.prefill(&input_ids).unwrap();
    
    // Decode
    let mut generated_ids = Vec::new();
    let mut current_token = *prefill_output.last().unwrap();
    
    for i in 0..max_tokens {
        let decode_config = ForwardPassConfig {
            is_prefill: false,
            batch_size: 1,
            seq_len: 1,
            cache_len: (input_ids.len() + i) as i32,
            temperature,
            seed,
        };
        
        current_token = adapter.decode_token(current_token).unwrap();
        generated_ids.push(current_token);
        
        if current_token == adapter.encoder.as_ref().unwrap().eos_token_id() {
            break;
        }
    }
    
    adapter.unload().unwrap();
    
    generated_ids
}

fn log_token_differences(run0: &[u32], run_n: &[u32]) {
    let min_len = run0.len().min(run_n.len());
    
    for i in 0..min_len {
        if run0[i] != run_n[i] {
            tracing::error!("    Position {}: {} != {}", i, run0[i], run_n[i]);
        }
    }
    
    if run0.len() != run_n.len() {
        tracing::error!("    Length mismatch: {} != {}", run0.len(), run_n.len());
    }
}
```

### Reproducibility Report
```markdown
# Llama Reproducibility Report

**Date**: 2025-10-04  
**Models**: Qwen2.5-0.5B, Phi-3-mini-4k  
**Test**: 10 runs per prompt with fixed seed

## Qwen2.5-0.5B Results

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

### Test 4: "In a galaxy far, far away" (seed=42)
- **Runs**: 10
- **Result**: ✅ PASS (all identical)

### Test 5: "The meaning of life is" (seed=123)
- **Runs**: 10
- **Result**: ✅ PASS (all identical)

**Qwen Summary**: 5/5 tests passed (100% reproducibility)

---

## Phi-3-mini-4k Results

### Test 1: "Write a haiku about ocean waves" (seed=42)
- **Runs**: 10
- **Result**: ✅ PASS (all identical)
- **Token sequence**: [22557, 263, 447, 29874, 12356, 29871, 18350, ...]
- **Output**: "Ocean waves crash / Against the rocky shoreline / Eternal rhythm"

### Test 2-5: ...
- **Result**: ✅ PASS (all identical)

**Phi-3 Summary**: 5/5 tests passed (100% reproducibility)

---

## Overall Summary

- **Total tests**: 10 (5 Qwen + 5 Phi-3)
- **Passed**: 10
- **Failed**: 0
- **Reproducibility**: 100%

## Conclusion

Both Qwen2.5-0.5B and Phi-3-mini-4k demonstrate perfect reproducibility with seeded RNG. All 10 runs per test produce identical token sequences, confirming deterministic generation.
```

---

## Testing Strategy

### Reproducibility Tests
- Run 10 tests for Qwen (5 prompts × 10 runs each)
- Run 10 tests for Phi-3 (5 prompts × 10 runs each)
- Verify 100% reproducibility

### Validation Tests
- Test with different seeds (42, 123, 999)
- Test with different temperatures (0.5, 0.7, 0.9)
- Test with different prompt lengths

### Manual Verification
1. Run full reproducibility test suite
2. Verify all tests pass
3. Review reproducibility report
4. Check logs for any warnings

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed
- [ ] All reproducibility tests passing (100%)
- [ ] Reproducibility report generated
- [ ] Documentation updated
- [ ] Story marked complete in day-tracker.md

---

## References

- Spec: `bin/.specs/01_M0_worker_orcd.md` Section 6.9 (Reproducibility)
- Related Stories: LT-035, LT-026, FT-020

---

**Status**: Ready for execution  
**Owner**: Llama-Beta  
**Created**: 2025-10-04

---

Detailed by Project Management Team — ready to implement 📋
