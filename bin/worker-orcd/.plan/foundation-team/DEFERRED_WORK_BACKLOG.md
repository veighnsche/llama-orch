# Deferred Work Backlog

**Team**: Foundation-Alpha  
**Created**: 2025-10-04  
**Status**: Planning Document  
**Context**: Post-Sprint 3 deferred features

---

## Overview

This document tracks all work deferred from Sprint 3 (Shared Kernels) and provides a structured plan for implementation in future sprints. All deferred items are **post-M0** and should be prioritized after M0 validation is complete.

**Total Deferred Work**: ~5-7 days (1 week sprint)

---

## Deferred Features Summary

| Feature | Complexity | Estimate | Priority | Sprint Target |
|---------|-----------|----------|----------|---------------|
| Top-P (Nucleus) Sampling | High | 1-2 days | High | Sprint 4 |
| Top-K Sampling | Medium | 1 day | High | Sprint 4 |
| Repetition Penalty | Medium | 1 day | Medium | Sprint 4 |
| Stop Sequences | High | 1-2 days | High | Sprint 4 |
| Min-P Sampling | Low | 0.5 days | Low | Sprint 5 (optional) |
| HTTP API Extension | Low | 0.5 days | High | Sprint 4 |

**Total**: 5-7 days for Sprint 4 (Advanced Sampling Parameters)

---

## Story 1: FT-019-Extended - Advanced Sampling Parameters

**Title**: Advanced Sampling Parameters  
**Size**: L (5-7 days)  
**Priority**: High  
**Sprint**: Sprint 4 (Post-M0)  
**Spec Ref**: M0-W-1421, GENERATION_PARAMETERS_ANALYSIS.md

### Story Description

Implement advanced sampling parameters (top-p, top-k, repetition penalty, stop sequences) to achieve competitive parity with OpenAI/llama.cpp/LM Studio. These parameters improve generation quality and enable structured output.

### Acceptance Criteria

**Top-P (Nucleus Sampling)**:
- [ ] Kernel sorts logits in descending order
- [ ] Computes cumulative probability
- [ ] Filters tokens where cumsum > top_p
- [ ] Re-normalizes remaining probabilities
- [ ] Unit tests validate filtering (5+ tests)
- [ ] Integration tests with temperature + top-p

**Top-K Sampling**:
- [ ] Kernel performs partial sort to find top K
- [ ] Zeros out logits outside top K
- [ ] Re-normalizes remaining probabilities
- [ ] Unit tests validate filtering (4+ tests)
- [ ] Integration tests with temperature + top-k

**Repetition Penalty**:
- [ ] History buffer tracks generated tokens
- [ ] Kernel applies penalty to tokens in history
- [ ] Penalty formula: `logits[i] /= penalty` (if positive) or `logits[i] *= penalty` (if negative)
- [ ] Unit tests validate penalty application (4+ tests)
- [ ] Integration tests with generation loop

**Stop Sequences**:
- [ ] Tokenize stop strings (up to 4 sequences)
- [ ] Pattern matching against generated sequence
- [ ] Early termination on match
- [ ] Unit tests validate matching (5+ tests)
- [ ] Integration tests with generation loop

**Min-P Sampling** (Optional):
- [ ] Kernel filters tokens below min_p threshold
- [ ] Re-normalizes remaining probabilities
- [ ] Unit tests validate filtering (3+ tests)

**HTTP API Extension**:
- [ ] Add `top_p` parameter (0.0-1.0, default 1.0)
- [ ] Add `top_k` parameter (0-vocab_size, default 0)
- [ ] Add `repetition_penalty` parameter (0.0-2.0, default 1.0)
- [ ] Add `stop` parameter (array of strings, max 4)
- [ ] Add `min_p` parameter (0.0-1.0, default 0.0)
- [ ] Validation for all parameters
- [ ] Backward compatibility (old requests work)

### Implementation Phases

**Phase 1: Top-K + Top-P** (3 days)
- Day 1: Sorting infrastructure (thrust::sort or custom)
- Day 2: Top-K filtering kernel + tests
- Day 3: Top-P filtering kernel + tests

**Phase 2: Repetition Penalty** (1 day)
- Day 4: History tracking + penalty kernel + tests

**Phase 3: Stop Sequences** (2 days)
- Day 5: Tokenization + pattern matching
- Day 6: Integration + tests

**Phase 4: Min-P + Integration** (1 day)
- Day 7: Min-P kernel + HTTP API extension + backward compatibility tests

### Technical Details

**Files to Create/Modify**:
- `cuda/kernels/sampling.cu` - Add advanced sampling kernels
- `cuda/kernels/sampling.cuh` - Add advanced sampling interface
- `cuda/include/sampling_config.h` - Add SamplingConfig struct
- `cuda/tests/test_sampling.cu` - Add advanced sampling tests
- `src/http/execute.rs` - Extend request schema
- `src/cuda/inference.rs` - Wire advanced parameters

**Key Interfaces**:
```cpp
struct SamplingConfig {
    float temperature = 1.0f;
    float top_p = 1.0f;           // 1.0 = disabled
    int top_k = 0;                // 0 = disabled
    float repetition_penalty = 1.0f;  // 1.0 = disabled
    float min_p = 0.0f;           // 0.0 = disabled
    
    // Stop sequences (tokenized)
    const int* stop_sequences[4] = {nullptr, nullptr, nullptr, nullptr};
    int stop_sequence_lengths[4] = {0, 0, 0, 0};
};

int launch_advanced_sample(
    const float* logits,
    int vocab_size,
    const SamplingConfig& config,
    const int* history,
    int history_length,
    float random_value,
    cudaStream_t stream = 0
);
```

### Dependencies

**Upstream (Required)**:
- ✅ M0 complete (core sampling working)
- ✅ HTTP API stable
- ✅ Testing infrastructure in place

**Downstream (Unblocks)**:
- Production-quality generation
- Structured output (JSON, code)
- Competitive parity with other APIs

### Testing Strategy

**Unit Tests** (25+ tests):
- Top-K: 5 tests (basic, edge cases, large K, small K, K > vocab_size)
- Top-P: 5 tests (basic, edge cases, p=0.0, p=1.0, numerical stability)
- Repetition penalty: 4 tests (basic, no history, full history, edge cases)
- Stop sequences: 5 tests (single match, multiple sequences, partial match, no match, edge cases)
- Min-P: 3 tests (basic, edge cases, numerical stability)
- Combined: 3 tests (top-k + top-p, all parameters, backward compatibility)

**Integration Tests** (5+ tests):
- Temperature + top-p + top-k pipeline
- Repetition penalty with generation loop
- Stop sequences with generation loop
- Backward compatibility (old requests work)
- Parameter validation (reject invalid values)

### Definition of Done

- [ ] All acceptance criteria met
- [ ] Code reviewed (self-review)
- [ ] Unit tests passing (25+ tests)
- [ ] Integration tests passing (5+ tests)
- [ ] HTTP API extended and documented
- [ ] Backward compatibility verified
- [ ] Performance profiling complete
- [ ] Documentation updated

---

## Story 2: FT-021 - FP16 Sampling Support

**Title**: FP16 Sampling Support  
**Size**: S (1 day)  
**Priority**: Medium  
**Sprint**: Sprint 5 (Post-M0)  
**Spec Ref**: M0-W-1421

### Story Description

Add FP16 support to all sampling kernels (greedy, stochastic, advanced). Currently only FP32 is supported. FP16 reduces memory bandwidth and improves performance.

### Acceptance Criteria

- [ ] FP16 greedy sampling kernel
- [ ] FP16 softmax kernel
- [ ] FP16 advanced sampling kernels (top-k, top-p, etc.)
- [ ] Unit tests for FP16 variants (10+ tests)
- [ ] Performance comparison (FP16 vs FP32)

### Implementation Details

**Files to Modify**:
- `cuda/kernels/sampling.cu` - Add FP16 variants
- `cuda/kernels/sampling.cuh` - Add FP16 declarations
- `cuda/tests/test_sampling.cu` - Add FP16 tests

**Key Functions**:
```cpp
int launch_greedy_sample_fp16(const half* logits, int vocab_size, cudaStream_t stream);
int launch_stochastic_sample_fp16(const half* logits, int vocab_size, float random_value, cudaStream_t stream);
```

### Dependencies

**Upstream**:
- FT-019-Extended complete (advanced sampling)

**Downstream**:
- Performance optimization (FP16 reduces bandwidth)

---

## Story 3: FT-022 - Optimized CDF Computation

**Title**: Optimized CDF Computation with Parallel Prefix Sum  
**Size**: M (2 days)  
**Priority**: Low  
**Sprint**: Sprint 6 (Post-M0)  
**Spec Ref**: M0-W-1421

### Story Description

Optimize CDF computation for stochastic sampling using parallel prefix sum (scan). Current implementation uses linear scan which is simple but not optimal for large vocabularies.

### Acceptance Criteria

- [ ] Parallel prefix sum kernel (scan)
- [ ] Binary search in CDF for sampling
- [ ] Performance improvement vs linear scan
- [ ] Unit tests validate correctness (5+ tests)
- [ ] Benchmark results documented

### Implementation Details

**Files to Modify**:
- `cuda/kernels/sampling.cu` - Add optimized CDF computation
- `cuda/kernels/sampling.cuh` - Add optimized interface
- `cuda/tests/test_sampling.cu` - Add performance tests

**Key Functions**:
```cpp
__global__ void compute_cdf_parallel(const float* probs, float* cdf, int vocab_size);
__global__ void sample_from_cdf_binary_search(const float* cdf, int vocab_size, float random_value, int* token_id);
```

### Dependencies

**Upstream**:
- FT-019-Extended complete (advanced sampling)

**Downstream**:
- Performance optimization for large vocabularies

### Performance Target

- **Current**: O(vocab_size) linear scan
- **Optimized**: O(log vocab_size) binary search after O(vocab_size) parallel prefix sum
- **Expected Improvement**: 2-5x faster for large vocabularies (>50k tokens)

---

## Implementation Roadmap

### Sprint 4: Advanced Sampling (Post-M0)
**Duration**: 1 week (5-7 days)  
**Focus**: Close competitive gap with OpenAI/llama.cpp

**Stories**:
- FT-019-Extended: Advanced Sampling Parameters (5-7 days)

**Deliverables**:
- Top-P (nucleus) sampling ✅
- Top-K sampling ✅
- Repetition penalty ✅
- Stop sequences ✅
- Min-P sampling ✅ (optional)
- HTTP API extension ✅

**Success Criteria**:
- All advanced parameters working
- Backward compatible
- Comprehensive test coverage
- Documentation complete

### Sprint 5: Performance & Polish (Post-M0)
**Duration**: 3-5 days  
**Focus**: Performance optimization and FP16 support

**Stories**:
- FT-021: FP16 Sampling Support (1 day)
- FT-023: Sampling Performance Profiling (1 day)
- FT-025: Documentation & Examples (1 day)

**Deliverables**:
- FP16 sampling kernels ✅
- Performance benchmarks ✅
- User documentation ✅
- Example requests ✅

### Sprint 6: Advanced Optimizations (Optional)
**Duration**: 2-3 days  
**Focus**: Advanced performance optimizations

**Stories**:
- FT-022: Optimized CDF Computation (2 days)
- FT-026: Fused Sampling Kernels (1 day, optional)

**Deliverables**:
- Parallel prefix sum CDF ✅
- Binary search sampling ✅
- Fused kernels (optional) ✅

---

## Prioritization Framework

### High Priority (Must Have for Production)
1. **Top-P + Top-K**: Most commonly used parameters
2. **Stop sequences**: Critical for structured output (JSON, code)
3. **HTTP API extension**: Required for client integration

### Medium Priority (Should Have)
4. **Repetition penalty**: Improves quality, but not critical
5. **FP16 support**: Performance optimization

### Low Priority (Nice to Have)
6. **Min-P sampling**: Rarely used
7. **Optimized CDF**: Performance optimization for large vocabularies
8. **Fused kernels**: Advanced optimization

---

## Risk Assessment

### Technical Risks

**Sorting Complexity** (High):
- GPU sorting is non-trivial
- Thrust library adds dependency
- Custom sorting requires careful implementation
- **Mitigation**: Use Thrust for initial implementation, optimize later if needed

**Parameter Interactions** (Medium):
- Top-K + Top-P can interact in unexpected ways
- Repetition penalty + temperature can conflict
- **Mitigation**: Comprehensive integration tests, clear precedence rules

**Performance Regression** (Medium):
- Sorting can be expensive (O(n log n))
- History tracking adds memory overhead
- **Mitigation**: Performance profiling, optimization pass

**Backward Compatibility** (Low):
- New parameters must not break old requests
- **Mitigation**: All parameters optional with defaults

### Timeline Risks

**Scope Creep** (Medium):
- Advanced features can expand beyond estimate
- **Mitigation**: Strict scope definition, MVP thinking

**Testing Overhead** (Medium):
- 25+ unit tests + 5+ integration tests
- **Mitigation**: Parallel test development, reuse test infrastructure

---

## Success Metrics

### Functional Metrics
- ✅ All advanced parameters implemented
- ✅ All unit tests passing (25+ tests)
- ✅ All integration tests passing (5+ tests)
- ✅ Backward compatibility maintained

### Quality Metrics
- ✅ No regressions in core sampling
- ✅ Performance acceptable (sorting < 10% of total latency)
- ✅ Numerical stability maintained

### Competitive Metrics
- ✅ Parameter parity with OpenAI (8/10 parameters)
- ✅ Parameter parity with llama.cpp (8/12 parameters)
- ✅ All critical parameters implemented

---

## Implementation Guidelines

### Design Principles

1. **Backward Compatibility First**
   - All new parameters optional
   - Default values disable features
   - Old requests continue to work

2. **Performance Awareness**
   - Profile before optimizing
   - Sorting should not dominate latency
   - Use efficient algorithms (Thrust, parallel primitives)

3. **Test Coverage**
   - Unit test each parameter independently
   - Integration test parameter combinations
   - Verify backward compatibility

4. **Documentation**
   - Document parameter behavior
   - Provide example requests
   - Explain parameter interactions

### Code Organization

**Kernel Files**:
- `sampling.cu` - All sampling kernels (keep consolidated)
- `sampling.cuh` - Public interface
- `sampling_config.h` - Configuration struct

**Test Files**:
- `test_sampling.cu` - All sampling tests (keep consolidated)

**Integration Files**:
- `src/http/execute.rs` - HTTP request schema
- `src/cuda/inference.rs` - FFI integration

### Testing Strategy

**Unit Tests**:
- Test each parameter independently
- Test edge cases (boundary values)
- Test error handling (invalid values)
- Test numerical stability

**Integration Tests**:
- Test parameter combinations
- Test with temperature scaling
- Test with generation loop
- Test backward compatibility

**Performance Tests**:
- Benchmark sorting overhead
- Profile memory usage
- Compare FP16 vs FP32

---

## Dependencies & Blockers

### Prerequisites for FT-019-Extended

**Must Be Complete**:
- ✅ M0 validation (worker loads model, executes inference)
- ✅ Core sampling working (greedy + stochastic + RNG)
- ✅ HTTP API stable
- ✅ Testing infrastructure in place

**Should Be Complete**:
- ✅ Sprint 4 planning (story breakdown, estimates)
- ✅ Thrust library available (or custom sort implemented)

**Nice to Have**:
- Performance baseline established
- User feedback on core sampling

### Blockers

**None Currently Identified**

**Potential Blockers**:
- Thrust library compatibility issues → Use custom sort
- Performance unacceptable → Optimize or defer
- Scope too large → Split into multiple stories

---

## Alternative Approaches Considered

### 1. Implement All Parameters in Sprint 3
**Pros**: Feature-complete M0, competitive parity
**Cons**: 5-7 additional days, increased risk, scope creep
**Decision**: Rejected (too risky for M0)

### 2. Implement Only Top-P/Top-K
**Pros**: Most commonly used, moderate complexity
**Cons**: Still missing repetition penalty and stop sequences
**Decision**: Considered for Phase 1, but full implementation preferred

### 3. Use CPU-Side Sampling
**Pros**: Simpler implementation, no GPU sorting
**Cons**: Requires copying logits to CPU (slow), defeats GPU acceleration
**Decision**: Rejected (performance unacceptable)

### 4. Use Thrust for All Operations
**Pros**: Well-tested library, fast implementation
**Cons**: External dependency, less control
**Decision**: Accepted for initial implementation, custom kernels if needed

---

## Communication Plan

### Stakeholder Updates

**After M0 Complete**:
- Inform user that advanced parameters are deferred
- Provide workarounds for missing features
- Share implementation timeline (Sprint 4)

**During Sprint 4**:
- Weekly progress updates
- Demo advanced parameters as they're completed
- Gather user feedback on parameter behavior

**After Sprint 4 Complete**:
- Announce feature parity with competitors
- Provide migration guide for users
- Share performance benchmarks

---

## Lessons Learned (Proactive)

### From Sprint 3 Deferral

1. **Recognize Scope Creep Early**
   - Original estimate: 2 days
   - Expanded scope: 7+ days
   - Action: Split into MVP + advanced

2. **MVP Thinking**
   - Ship core functionality first
   - Iterate with advanced features
   - Validate with users before expanding

3. **Complexity Assessment**
   - Simple algorithms: implement immediately
   - Complex algorithms: defer and focus
   - Unknown complexity: spike first

4. **User Value Prioritization**
   - High value: implement in M0
   - Medium value: implement in M1
   - Low value: defer to M2+

### For Future Sprints

1. **Estimate Conservatively**
   - Add buffer for unknowns
   - Account for testing time
   - Include integration overhead

2. **Define MVP Clearly**
   - What's the minimum viable?
   - What can be deferred?
   - What's the 80/20 split?

3. **Communicate Early**
   - Share deferral decisions quickly
   - Explain rationale clearly
   - Provide alternative timeline

---

## References

- **Deferral Decision**: `ADVANCED_SAMPLING_DEFERRAL.md`
- **Original Story**: `todo/FT-019-stochastic-sampling.md`
- **Completion Summary**: `FT-019_COMPLETION_SUMMARY.md`
- **Spec**: `bin/.specs/01_M0_worker_orcd.md` (M0-W-1421)
- **Analysis**: `bin/.specs/.docs/GENERATION_PARAMETERS_ANALYSIS.md`

---

## Appendix: Feature Comparison

| Feature | M0 (Current) | M0 (Planned) | OpenAI | llama.cpp | LM Studio |
|---------|--------------|--------------|--------|-----------|-----------|
| **Core Parameters** |
| Temperature | ✅ | ✅ | ✅ | ✅ | ✅ |
| Max Tokens | ✅ | ✅ | ✅ | ✅ | ✅ |
| Seed | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Advanced Parameters** |
| Top-P | ❌ | ✅ | ✅ | ✅ | ✅ |
| Top-K | ❌ | ✅ | ❌ | ✅ | ✅ |
| Repetition Penalty | ❌ | ✅ | ❌ | ✅ | ✅ |
| Stop Sequences | ❌ | ✅ | ✅ | ✅ | ✅ |
| Min-P | ❌ | ✅ | ❌ | ✅ | ❌ |
| **Totals** |
| Parameters | 3 | 8 | 10 | 12 | 13 |
| Coverage | 30% | 80% | 100% | 100% | 100% |

**Gap Analysis**:
- M0 (Current): 3/10 parameters (30% coverage)
- M0 (Planned): 8/10 parameters (80% coverage)
- Remaining gap: Frequency penalty, presence penalty (OpenAI-specific)

**Conclusion**: Post-Sprint 4, M0 will have 80% parameter coverage, sufficient for production use.

---
Built by Foundation-Alpha 🏗️
