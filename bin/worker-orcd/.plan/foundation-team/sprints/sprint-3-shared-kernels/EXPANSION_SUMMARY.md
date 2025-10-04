# Sprint 3 Expansion: Advanced Generation Parameters

**Date**: 2025-10-04  
**Status**: Planned  
**Impact**: +1 day to sprint (16 → 17 days)

---

## What Changed

### FT-019: Stochastic Sampling (EXPANDED)

**Before**:
- Size: M (2 days)
- Days: 34-35
- Scope: Basic stochastic sampling (softmax + sample from distribution)

**After**:
- Size: L (3 days)
- Days: 34-36
- Scope: Advanced sampling with competitive parity parameters

**Added Features**:
1. **Top-P (nucleus sampling)** — Filter tokens by cumulative probability
2. **Top-K sampling** — Keep only top K tokens
3. **Repetition penalty** — Penalize already-generated tokens
4. **Stop sequences** — Terminate on match (up to 4 sequences)
5. **Min-P sampling** — Minimum probability threshold (optional)
6. **HTTP API extension** — New parameters with validation
7. **Backward compatibility** — Old requests still work

---

## Why This Expansion

### Problem Identified

User question: *"What about all the other config that other engines expose to the API?"*

**Analysis findings** (see `bin/.specs/.docs/GENERATION_PARAMETERS_ANALYSIS.md`):
- M0 originally had only **3 parameters** (temperature, max_tokens, seed)
- Industry standard: **10-13 parameters** (OpenAI: 10, llama.cpp: 12, LM Studio: 13)
- Missing critical parameters: top-p, top-k, stop sequences, repetition penalty

### Competitive Comparison

| Feature | M0 (before) | M0 (after) | OpenAI | llama.cpp | LM Studio |
|---------|-------------|------------|--------|-----------|-----------|
| **Total Parameters** | 3 | 8 | 10 | 12 | 13 |
| temperature | ✅ | ✅ | ✅ | ✅ | ✅ |
| max_tokens | ✅ | ✅ | ✅ | ✅ | ✅ |
| seed | ✅ | ✅ | ✅ (beta) | ✅ | ✅ |
| **top_p** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **top_k** | ❌ | ✅ | ❌ | ✅ | ✅ |
| **repetition_penalty** | ❌ | ✅ | ❌ | ✅ | ✅ |
| **stop sequences** | ❌ | ✅ | ✅ | ✅ | ✅ |
| frequency_penalty | ❌ | ❌ | ✅ | ❌ | ✅ |
| presence_penalty | ❌ | ❌ | ✅ | ❌ | ✅ |
| min_p | ❌ | ✅ (opt) | ❌ | ✅ | ✅ |

**Result**: M0 now has **8 parameters** (up from 3), achieving competitive parity with major APIs.

---

## Rationale

### Why Add Now (Sprint 3) vs Defer to M1?

**Efficiency**:
- Sampling infrastructure is already being built in FT-019
- Adding parameters now is incremental work (kernels + API)
- Deferring to M1 would require revisiting sampling code

**User Expectations**:
- Users coming from OpenAI/llama.cpp/LM Studio expect these parameters
- Stop sequences are critical for structured output (JSON, code)
- Top-p/top-k are industry standard

**Cost/Benefit**:
- Cost: +1 day to Sprint 3 (16 → 17 days)
- Benefit: Eliminates need for separate M1 story (saves 1-2 weeks later)
- Net: More efficient to add now

---

## Impact on Sprint 3

### Timeline Changes

| Story | Before | After | Change |
|-------|--------|-------|--------|
| FT-019 | Days 34-35 (M, 2 days) | Days 34-36 (L, 3 days) | +1 day |
| FT-020 | Day 36 | Day 37 | Shifted +1 day |
| **Sprint Total** | **Days 23-38 (16 days)** | **Days 23-39 (17 days)** | **+1 day** |

### Sprint 4 Impact

- Sprint 4 (Integration + Gate 1) now starts Day 39 (was Day 38)
- No other changes to downstream sprints

---

## Technical Details

### New CUDA Kernels

1. `apply_top_k()` — Filter to top K tokens
2. `apply_top_p()` — Nucleus sampling filter
3. `apply_repetition_penalty()` — Penalize history tokens
4. `apply_min_p()` — Minimum probability filter
5. `check_stop_sequences()` — Early termination check

### HTTP API Extension

**New Request Parameters**:
```json
{
  "job_id": "job-xyz",
  "prompt": "Write a haiku",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,           // NEW
  "top_k": 50,            // NEW
  "repetition_penalty": 1.1,  // NEW
  "stop": ["\n\n", "END"],    // NEW
  "seed": 42
}
```

**Validation**:
- `top_p`: 0.0-1.0, default 1.0 (disabled)
- `top_k`: 0-vocab_size, default 0 (disabled)
- `repetition_penalty`: 0.0-2.0, default 1.0 (disabled)
- `stop`: array of strings, max 4 sequences

**Backward Compatibility**: All new parameters are optional. Old requests (temperature only) continue to work.

---

## Testing Expansion

### Additional Unit Tests (6 new tests)

1. Top-K filtering correctness
2. Top-P nucleus sampling correctness
3. Repetition penalty application
4. Stop sequence detection
5. Min-P filtering
6. Combined parameters (top-k + top-p + repetition_penalty)

### Additional Integration Tests (3 new tests)

1. HTTP API with all new parameters
2. Backward compatibility (old requests work)
3. Parameter validation (reject invalid values)

**Total Tests**: 10+ unit tests, 5+ integration tests (up from 4 unit, 2 integration)

---

## References

- **Analysis Document**: `bin/.specs/.docs/GENERATION_PARAMETERS_ANALYSIS.md`
- **Updated Story**: `todo/FT-019-stochastic-sampling.md`
- **Sprint README**: `README.md` (updated with new timeline)
- **M0 Spec**: `bin/.specs/01_M0_worker_orcd.md` §7.1 (HTTP API)

---

## Approval

**Decision**: Expand FT-019 to include advanced generation parameters in Sprint 3.

**Approved by**: User (2025-10-04)

**Rationale**: Achieves competitive parity with industry APIs, meets user expectations, and is more efficient than deferring to M1.

---

**Status**: ✅ Planned and documented  
**Next Steps**: Execute FT-019 (Days 34-36) with expanded scope
