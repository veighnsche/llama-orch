# Sprint 4: Advanced Sampling Parameters

**Team**: Foundation-Alpha  
**Duration**: 1 week (5-7 days)  
**Start**: Post-M0 validation  
**Status**: 📋 Planned  
**Goal**: Close competitive gap with OpenAI/llama.cpp

---

## Sprint Overview

Implement advanced sampling parameters deferred from Sprint 3 to achieve competitive parity with industry-standard LLM APIs. This sprint focuses on generation quality improvements and structured output support.

**Why This Sprint**: M0 proved core sampling works. Now add advanced features that users expect from production LLM APIs.

---

## Sprint Goals

### Primary Goals
1. ✅ Implement Top-P (nucleus) sampling
2. ✅ Implement Top-K sampling
3. ✅ Implement repetition penalty
4. ✅ Implement stop sequences
5. ✅ Extend HTTP API with new parameters

### Secondary Goals
6. ✅ Implement Min-P sampling (optional)
7. ✅ Comprehensive integration tests
8. ✅ Performance profiling

### Stretch Goals
9. ⏸️ FP16 sampling support (defer to Sprint 5 if time-constrained)
10. ⏸️ Optimized CDF computation (defer to Sprint 6 if time-constrained)

---

## Stories

### FT-019-Extended: Advanced Sampling Parameters
**Size**: L (5-7 days)  
**Priority**: P0 (High)  
**Owner**: Foundation-Alpha

**Breakdown**:
- Day 1: Top-K sorting infrastructure + kernel
- Day 2: Top-K tests + Top-P kernel
- Day 3: Top-P tests + integration
- Day 4: Repetition penalty + tests
- Day 5: Stop sequences implementation
- Day 6: Stop sequences tests + HTTP API extension
- Day 7: Integration tests + documentation

**Deliverables**:
- 5 new sampling kernels
- 25+ unit tests
- 5+ integration tests
- HTTP API extension
- Updated documentation

---

## Technical Architecture

### Sampling Pipeline (Extended)

```
Input: Logits [vocab_size]
  ↓
[1] Temperature Scaling (if temp != 1.0)
  ↓
[2] Repetition Penalty (if penalty != 1.0 && history exists)
  ↓
[3] Top-K Filtering (if top_k > 0)
  ↓
[4] Top-P Filtering (if top_p < 1.0)
  ↓
[5] Min-P Filtering (if min_p > 0.0)
  ↓
[6] Softmax (convert to probabilities)
  ↓
[7] Sample from distribution (using RNG)
  ↓
Output: Token ID
  ↓
[8] Check Stop Sequences (if configured)
  ↓
Output: Token ID + should_stop flag
```

### New Kernels

**Filtering Kernels**:
```cpp
__global__ void apply_top_k(float* logits, int vocab_size, int top_k);
__global__ void apply_top_p(float* logits, int vocab_size, float top_p);
__global__ void apply_repetition_penalty(float* logits, int vocab_size, const int* history, int history_length, float penalty);
__global__ void apply_min_p(float* logits, int vocab_size, float min_p);
```

**Utility Kernels**:
```cpp
__global__ void partial_sort_descending(float* logits, int* indices, int vocab_size, int k);
__global__ void compute_cumulative_probs(const float* probs, float* cumsum, int vocab_size);
```

**Host Functions**:
```cpp
bool check_stop_sequences(const int* generated_tokens, int num_generated, const SamplingConfig& config);
```

### Configuration Struct

```cpp
struct SamplingConfig {
    // Core parameters
    float temperature = 1.0f;
    uint64_t seed = 0;
    
    // Advanced parameters
    float top_p = 1.0f;           // 1.0 = disabled
    int top_k = 0;                // 0 = disabled
    float repetition_penalty = 1.0f;  // 1.0 = disabled
    float min_p = 0.0f;           // 0.0 = disabled
    
    // Stop sequences (tokenized)
    const int* stop_sequences[4] = {nullptr, nullptr, nullptr, nullptr};
    int stop_sequence_lengths[4] = {0, 0, 0, 0};
    
    // Validation
    bool is_valid() const {
        return temperature >= 0.0f && temperature <= 2.0f
            && top_p >= 0.0f && top_p <= 1.0f
            && top_k >= 0
            && repetition_penalty >= 0.0f && repetition_penalty <= 2.0f
            && min_p >= 0.0f && min_p <= 1.0f;
    }
};
```

---

## HTTP API Extension

### Extended Request Schema

```json
{
  "job_id": "job-xyz",
  "prompt": "Write a haiku about GPU computing",
  "max_tokens": 100,
  
  // Core parameters (existing)
  "temperature": 0.7,
  "seed": 42,
  
  // Advanced parameters (new)
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.1,
  "stop": ["\n\n", "END"],
  "min_p": 0.05
}
```

### Validation Rules

| Parameter | Type | Range | Default | Required |
|-----------|------|-------|---------|----------|
| temperature | float | 0.0-2.0 | 1.0 | No |
| seed | uint64 | 0-UINT64_MAX | auto | No |
| top_p | float | 0.0-1.0 | 1.0 | No |
| top_k | int | 0-vocab_size | 0 | No |
| repetition_penalty | float | 0.0-2.0 | 1.0 | No |
| stop | array[string] | max 4 sequences | [] | No |
| min_p | float | 0.0-1.0 | 0.0 | No |

**Validation Logic**:
```rust
fn validate_sampling_params(req: &ExecuteRequest) -> Result<(), ValidationError> {
    if req.temperature < 0.0 || req.temperature > 2.0 {
        return Err(ValidationError::InvalidTemperature);
    }
    if let Some(top_p) = req.top_p {
        if top_p < 0.0 || top_p > 1.0 {
            return Err(ValidationError::InvalidTopP);
        }
    }
    // ... etc
    Ok(())
}
```

---

## Testing Plan

### Unit Tests (25+ tests)

**Top-K Tests** (5 tests):
1. Basic top-k filtering (k=50, vocab=1000)
2. Edge case: k=0 (disabled)
3. Edge case: k=vocab_size (no filtering)
4. Edge case: k > vocab_size (clamp to vocab_size)
5. Large vocabulary (k=100, vocab=151936)

**Top-P Tests** (5 tests):
1. Basic top-p filtering (p=0.9)
2. Edge case: p=0.0 (keep only max)
3. Edge case: p=1.0 (disabled)
4. Numerical stability with large logits
5. Large vocabulary (p=0.9, vocab=151936)

**Repetition Penalty Tests** (4 tests):
1. Basic penalty application (penalty=1.1)
2. No history (penalty has no effect)
3. Full history (all tokens penalized)
4. Edge case: penalty=1.0 (disabled)

**Stop Sequences Tests** (5 tests):
1. Single sequence match
2. Multiple sequences (match first)
3. Partial match (no stop)
4. No match
5. Edge case: empty stop sequences

**Min-P Tests** (3 tests):
1. Basic filtering (min_p=0.05)
2. Edge case: min_p=0.0 (disabled)
3. Edge case: min_p=1.0 (keep only max)

**Combined Tests** (3 tests):
1. Top-K + Top-P together
2. All parameters together
3. Backward compatibility (no new parameters)

### Integration Tests (5+ tests)

1. **Temperature + Top-P + Top-K pipeline**
   - Verify filters applied in correct order
   - Verify sampling follows filtered distribution

2. **Repetition penalty with generation loop**
   - Generate 50 tokens with penalty=1.2
   - Verify reduced repetition vs no penalty

3. **Stop sequences with generation loop**
   - Generate until stop sequence matched
   - Verify early termination

4. **Backward compatibility**
   - Old request format (no new parameters)
   - Verify identical behavior to Sprint 3

5. **Parameter validation**
   - Invalid values rejected with clear errors
   - Valid values accepted

---

## Performance Targets

### Latency Budget

**Total Sampling Latency**: <5ms per token (target)

**Breakdown**:
- Temperature scaling: <0.1ms
- Repetition penalty: <0.5ms
- Top-K sorting: <2ms (most expensive)
- Top-P filtering: <1ms
- Min-P filtering: <0.1ms
- Softmax: <0.5ms
- Sampling: <0.1ms
- Stop sequence check: <0.1ms

**Total**: ~4.4ms (within budget)

**If Over Budget**: Optimize sorting (use Thrust, custom kernels, or parallel primitives)

### Memory Budget

**Additional VRAM**:
- History buffer: ~4 KB (1000 tokens × 4 bytes)
- Stop sequences: ~512 bytes (4 sequences × 32 tokens × 4 bytes)
- Temporary sort buffer: ~600 KB (151936 tokens × 4 bytes)

**Total**: ~605 KB (negligible compared to model size)

---

## Risk Mitigation

### Risk 1: Sorting Performance
**Risk**: GPU sorting may be too slow for large vocabularies

**Mitigation**:
- Use Thrust library (optimized)
- Profile early (Day 1)
- If too slow: implement partial sort (only find top K, don't sort all)
- Fallback: CPU-side sorting (copy logits, sort, copy back)

### Risk 2: Parameter Interactions
**Risk**: Parameters may interact in unexpected ways

**Mitigation**:
- Define clear precedence order
- Test all combinations
- Document interaction behavior
- Add integration tests for common combinations

### Risk 3: Backward Compatibility
**Risk**: New parameters may break old clients

**Mitigation**:
- All parameters optional
- Default values disable features
- Test old request format explicitly
- Version API if needed

### Risk 4: Scope Creep
**Risk**: Sprint may expand beyond 7 days

**Mitigation**:
- Strict scope definition (no new features)
- Min-P is optional (can be skipped)
- Daily progress tracking
- Cut scope if falling behind (defer Min-P)

---

## Success Criteria

### Functional Success
- ✅ All 5 advanced parameters implemented (or 4 if Min-P deferred)
- ✅ HTTP API extended with new parameters
- ✅ All unit tests passing (25+ tests)
- ✅ All integration tests passing (5+ tests)
- ✅ Backward compatibility maintained

### Quality Success
- ✅ No regressions in core sampling
- ✅ Performance within budget (<5ms per token)
- ✅ Numerical stability maintained
- ✅ Error handling comprehensive

### Competitive Success
- ✅ 8/10 parameters vs OpenAI (80% parity)
- ✅ 8/12 parameters vs llama.cpp (67% parity)
- ✅ All critical parameters implemented

---

## Daily Breakdown

### Day 1: Top-K Infrastructure
- Morning: Thrust integration + sorting kernel
- Afternoon: Top-K filtering kernel + basic tests
- EOD: 2 tests passing

### Day 2: Top-K Complete + Top-P Start
- Morning: Top-K edge case tests (3 more tests)
- Afternoon: Top-P kernel + cumulative probability
- EOD: Top-K complete (5 tests), Top-P kernel ready

### Day 3: Top-P Complete + Integration
- Morning: Top-P tests (5 tests)
- Afternoon: Integration tests (top-k + top-p together)
- EOD: Top-K + Top-P complete

### Day 4: Repetition Penalty
- Morning: History buffer + penalty kernel
- Afternoon: Tests (4 tests) + integration
- EOD: Repetition penalty complete

### Day 5: Stop Sequences Implementation
- Morning: Tokenization + pattern matching
- Afternoon: Integration with generation loop
- EOD: Stop sequences working (no tests yet)

### Day 6: Stop Sequences Tests + HTTP API
- Morning: Stop sequence tests (5 tests)
- Afternoon: HTTP API extension + validation
- EOD: Stop sequences complete, API extended

### Day 7: Min-P + Final Integration
- Morning: Min-P kernel + tests (3 tests)
- Afternoon: Backward compatibility tests + documentation
- EOD: Sprint complete, all features working

---

## Deliverables

### Code Artifacts
- [ ] 5 new sampling kernels (top-k, top-p, repetition, stop, min-p)
- [ ] Extended HTTP API schema
- [ ] Parameter validation logic
- [ ] 25+ unit tests
- [ ] 5+ integration tests

### Documentation Artifacts
- [ ] Kernel documentation (sampling.cuh)
- [ ] HTTP API documentation (updated)
- [ ] Parameter interaction guide
- [ ] Example requests
- [ ] Performance benchmarks

### Testing Artifacts
- [ ] Unit test report
- [ ] Integration test report
- [ ] Performance profiling results
- [ ] Backward compatibility verification

---

## Dependencies

### Prerequisites
- ✅ M0 complete and validated
- ✅ Core sampling working (FT-017, FT-018, FT-019, FT-020)
- ✅ HTTP API stable
- ✅ Testing infrastructure in place

### External Dependencies
- Thrust library (CUDA toolkit)
- Google Test (already in use)

---

## References

- **Backlog**: `../DEFERRED_WORK_BACKLOG.md`
- **Deferral Decision**: `../sprint-3-shared-kernels/ADVANCED_SAMPLING_DEFERRAL.md`
- **Original Story**: `../sprint-3-shared-kernels/todo/FT-019-stochastic-sampling.md`
- **Spec**: `bin/.specs/01_M0_worker_orcd.md` (M0-W-1421)

---
Built by Foundation-Alpha 🏗️
