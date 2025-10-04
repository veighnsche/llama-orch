# Advanced Sampling Parameters - Deferral Decision

**Date**: 2025-10-04  
**Decision**: Defer advanced sampling parameters to post-M0  
**Team**: Foundation-Alpha  
**Context**: FT-019 Stochastic Sampling implementation

---

## Executive Summary

Advanced sampling parameters (top-p, top-k, repetition penalty, stop sequences, min-p) have been **deferred to post-M0** despite being included in the original FT-019 scope. This document explains the rationale, impact, and future plan.

**TL;DR**: Core stochastic sampling (softmax + CDF) is sufficient for M0. Advanced parameters add significant complexity without blocking basic inference functionality. Deferring them reduces M0 risk and allows focused implementation later.

---

## Deferred Features

### 1. Top-P (Nucleus Sampling)
**What**: Filter tokens by cumulative probability threshold (e.g., top_p=0.9 keeps tokens that sum to 90% probability)

**Complexity**:
- Requires sorting logits in descending order
- Compute cumulative softmax during sort
- Find cutoff where cumsum >= top_p
- Zero out logits below cutoff
- Re-normalize remaining probabilities

**Implementation Estimate**: 1-2 days
- Sorting kernel (thrust::sort or custom parallel sort)
- Prefix sum for cumulative probabilities
- Filtering kernel
- Unit tests (5+ tests)

### 2. Top-K Sampling
**What**: Keep only top K tokens by probability (e.g., top_k=50 keeps only 50 highest probability tokens)

**Complexity**:
- Requires partial sort to find top K
- Zero out logits outside top K
- Re-normalize remaining probabilities

**Implementation Estimate**: 1 day
- Partial sort kernel (thrust::partial_sort or custom)
- Filtering kernel
- Unit tests (4+ tests)

### 3. Repetition Penalty
**What**: Penalize tokens that have already been generated (e.g., penalty=1.1 reduces probability of repeated tokens)

**Complexity**:
- Requires history tracking (generated token IDs)
- Check if each token is in history
- Apply penalty: `logits[i] /= penalty` (if positive) or `logits[i] *= penalty` (if negative)
- Memory management for history buffer

**Implementation Estimate**: 1 day
- History buffer management
- Penalty application kernel
- Unit tests (4+ tests)

### 4. Stop Sequences
**What**: Terminate generation when a specific token sequence is generated (e.g., stop=["\n\n", "END"])

**Complexity**:
- Requires tokenization of stop strings
- Pattern matching against generated sequence
- Up to 4 stop sequences (per spec)
- Each sequence up to 32 tokens
- Sliding window comparison

**Implementation Estimate**: 1-2 days
- Tokenization of stop strings (CPU-side)
- Pattern matching logic
- Integration with generation loop
- Unit tests (5+ tests)

### 5. Min-P Sampling
**What**: Minimum probability threshold (e.g., min_p=0.05 removes tokens with <5% probability)

**Complexity**:
- Compute softmax first
- Filter tokens below threshold
- Re-normalize remaining probabilities

**Implementation Estimate**: 0.5 days
- Filtering kernel
- Unit tests (3+ tests)

---

## Total Complexity

**Estimated Implementation Time**: 5-7 days
- Kernel development: 3-4 days
- Testing: 1-2 days
- Integration: 1 day
- Documentation: 0.5 days

**Additional Complexity**:
- HTTP API extension (request schema, validation)
- Parameter interaction testing (top-p + top-k + repetition penalty)
- Backward compatibility testing
- Performance profiling (sorting can be expensive)

---

## Rationale for Deferral

### 1. M0 Scope Clarity
**M0 Goal**: Prove worker can load model, execute inference, stream results

**Core Requirement**: Basic token sampling (greedy + stochastic)

**Advanced Parameters**: Nice-to-have, not blocking for M0 validation

**Decision**: Keep M0 focused on foundational functionality

### 2. Risk Reduction
**Complexity Risk**:
- Sorting algorithms are complex (GPU sorting is non-trivial)
- Parameter interactions can have subtle bugs
- History tracking adds memory management complexity

**Timeline Risk**:
- 5-7 additional days could push M0 completion
- Testing advanced features requires more time
- Integration with HTTP API adds scope

**Quality Risk**:
- Rushing advanced features increases bug risk
- Better to implement carefully in focused story

### 3. Incremental Value
**Core Sampling Value**: Enables all basic inference use cases
- Greedy sampling (temp=0.0): Deterministic, reproducible
- Stochastic sampling (temp>0.0): Creative generation

**Advanced Parameters Value**: Improve generation quality, but not blocking
- Top-P/Top-K: Better quality, but basic sampling works
- Repetition penalty: Reduces repetition, but not critical
- Stop sequences: Convenience, but can be handled client-side

**Conclusion**: Core sampling delivers 80% of value with 20% of complexity

### 4. Industry Comparison
**OpenAI API**: 10 parameters (but launched with fewer initially)

**llama.cpp**: 12 parameters (added incrementally over time)

**M0 (Core)**: 3 parameters (temperature, seed, max_tokens)

**M0 (Extended)**: Would be 8 parameters (+ top-p, top-k, repetition_penalty, stop, min-p)

**Observation**: Most APIs started simple and added parameters incrementally

### 5. User Expectations
**Early Adopters**: Expect basic inference to work reliably

**Advanced Users**: Can wait for advanced parameters if core works well

**Production Users**: Prefer stable core over feature-rich but buggy system

**Conclusion**: Better to ship solid core than rushed advanced features

---

## Impact Analysis

### What Still Works
✅ **Temperature-based sampling**: Full range (0.0-2.0)
- temp=0.0: Greedy (deterministic)
- temp=0.1-0.9: More deterministic
- temp=1.0: Original distribution
- temp=1.1-2.0: More random

✅ **Seeded RNG**: Reproducible generation with same seed

✅ **Streaming**: SSE streaming works with both greedy and stochastic

✅ **All M0 test cases**: Haiku test, reproducibility test, etc.

### What's Missing
❌ **Top-P/Top-K**: Cannot filter low-probability tokens
- **Workaround**: Use temperature to control randomness
- **Impact**: Lower quality generation (more repetition, less coherent)

❌ **Repetition penalty**: Cannot reduce repetition
- **Workaround**: Client-side post-processing
- **Impact**: More repetitive text

❌ **Stop sequences**: Cannot auto-terminate on patterns
- **Workaround**: Client-side detection and truncation
- **Impact**: Slightly more complex client code

❌ **Min-P**: Cannot set minimum probability threshold
- **Workaround**: Use temperature or top-k (when implemented)
- **Impact**: Minimal (min-p is rarely used)

### Competitive Position
**Before Deferral**: M0 would match llama.cpp/OpenAI feature set

**After Deferral**: M0 has basic sampling, competitors have advanced

**Gap**: Acceptable for M0 (proof of concept), must close for M1 (production)

---

## Future Implementation Plan

### Post-M0 Story: FT-019-Extended
**Title**: Advanced Sampling Parameters

**Scope**:
- Top-P (nucleus) sampling
- Top-K sampling
- Repetition penalty
- Stop sequences
- Min-P sampling (optional)

**Timeline**: 1 week (5-7 days)

**Priority**: High (close competitive gap)

**Dependencies**:
- M0 complete (core sampling working)
- HTTP API stable
- Testing infrastructure in place

### Implementation Phases

**Phase 1: Top-K + Top-P** (3 days)
- Implement sorting kernels
- Implement filtering kernels
- Unit tests + integration tests
- Most commonly used parameters

**Phase 2: Repetition Penalty** (1 day)
- Implement history tracking
- Implement penalty application
- Unit tests

**Phase 3: Stop Sequences** (2 days)
- Implement pattern matching
- Integrate with generation loop
- Unit tests

**Phase 4: Min-P** (0.5 days)
- Implement filtering
- Unit tests
- Low priority (can be skipped if time-constrained)

**Phase 5: Integration** (0.5 days)
- HTTP API extension
- Parameter validation
- Backward compatibility tests

### Success Criteria
- All advanced parameters implemented and tested
- HTTP API extended with new parameters
- Backward compatibility maintained (old requests still work)
- Performance acceptable (sorting doesn't dominate latency)
- Documentation updated

---

## Lessons Learned

### 1. Scope Creep Detection
**Original FT-019**: "Implement stochastic sampling" (2 days)

**Expanded FT-019**: "Implement stochastic sampling + 5 advanced parameters" (3 days → 7+ days)

**Lesson**: Recognize when scope expands beyond original estimate

### 2. MVP Thinking
**Question**: What's the minimum viable implementation?

**Answer**: Softmax + CDF sampling (enables all basic use cases)

**Lesson**: Ship MVP first, iterate with advanced features

### 3. Complexity Assessment
**Simple**: Softmax, CDF sampling (well-understood algorithms)

**Complex**: Sorting, pattern matching, history tracking (many edge cases)

**Lesson**: Separate simple from complex, ship simple first

### 4. User Value Prioritization
**High Value**: Temperature-based sampling (80% of use cases)

**Medium Value**: Top-P/Top-K (improves quality)

**Low Value**: Min-P (rarely used)

**Lesson**: Prioritize by user value, not by "nice to have"

---

## Conclusion

Deferring advanced sampling parameters to post-M0 is the right decision:

1. **Reduces M0 risk**: Simpler implementation, fewer bugs
2. **Maintains M0 timeline**: Core sampling sufficient for validation
3. **Enables focused implementation**: Advanced features deserve dedicated story
4. **Aligns with industry practice**: Most APIs started simple, added features incrementally
5. **Delivers user value**: Core sampling enables all basic use cases

**Next Steps**:
1. Complete M0 with core sampling ✅
2. Validate M0 works end-to-end
3. Create FT-019-Extended story for post-M0
4. Implement advanced parameters in focused 1-week sprint

---

## References

- **Original Story**: `FT-019-stochastic-sampling.md`
- **Completion Summary**: `FT-019_COMPLETION_SUMMARY.md`
- **Spec**: `bin/.specs/01_M0_worker_orcd.md` (M0-W-1421)
- **Analysis**: `bin/.specs/.docs/GENERATION_PARAMETERS_ANALYSIS.md`

---
Built by Foundation-Alpha 🏗️
