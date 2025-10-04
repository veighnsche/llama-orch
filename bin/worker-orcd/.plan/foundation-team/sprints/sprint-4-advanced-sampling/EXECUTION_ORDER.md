# Sprint 4: Advanced Sampling - Execution Order

**Team**: Foundation-Alpha  
**Sprint**: Sprint 4 - Advanced Sampling  
**Total Duration**: 7 days  
**Status**: 📋 Planning Complete

---

## Dependency Analysis

### Current State (Sprint 3 Complete)
✅ **FT-017**: Temperature scaling (Days 30-32)  
✅ **FT-018**: Greedy sampling (Day 33)  
✅ **FT-019**: Stochastic sampling core (Days 34-36)

**Available Infrastructure**:
- `temperature_scale_fp32()` - Temperature scaling kernel
- `softmax_fp32()` - Softmax with numerical stability
- `launch_greedy_sample()` - Greedy sampling (argmax)
- `launch_stochastic_sample()` - Basic stochastic sampling

**Missing Infrastructure** (Sprint 4 scope):
- Top-K filtering
- Top-P (nucleus) filtering
- Repetition penalty
- Stop sequences
- Min-P filtering
- HTTP API extensions

---

## Execution Order (Dependency-Driven)

### Phase 1: Core Filtering Kernels (Days 1-3)
**Story**: FT-019-EXT-1 (Top-K and Top-P)  
**Duration**: 3 days  
**Priority**: HIGH

**Why First**:
- Top-K and Top-P are the most commonly used parameters
- They are independent filters that modify logits
- Other parameters can be tested with these in place
- No dependencies on other advanced parameters

**Deliverables**:
- `apply_top_k()` kernel
- `apply_top_p()` kernel
- Unit tests (10+ tests: 5 top-k, 5 top-p)
- Integration tests (3 tests: combined usage)
- Performance profiling

**Dependencies**: None (uses existing softmax/sampling)

---

### Phase 2: Repetition Penalty (Day 4)
**Story**: FT-019-EXT-2 (Repetition Penalty)  
**Duration**: 1 day  
**Priority**: HIGH

**Why Second**:
- Requires history buffer management (new infrastructure)
- Independent of top-k/top-p (can be tested separately)
- Commonly used parameter (competitive parity)
- Builds history tracking needed for stop sequences

**Deliverables**:
- `apply_repetition_penalty()` kernel
- History buffer management (InferenceState class)
- Unit tests (4+ tests)
- Integration tests (2 tests: with temperature, with top-k/top-p)

**Dependencies**: 
- ✅ Basic sampling (Sprint 3)
- ⚠️ History buffer (new, created in this story)

---

### Phase 3: Stop Sequences (Days 5-6)
**Story**: FT-019-EXT-3 (Stop Sequences)  
**Duration**: 2 days  
**Priority**: CRITICAL

**Why Third**:
- Requires history buffer (created in Phase 2)
- Requires tokenization integration
- Critical for structured output (JSON, code)
- More complex than other parameters (pattern matching)

**Deliverables**:
- `check_stop_sequences()` function (CPU-side)
- Tokenization of stop strings
- Pattern matching against generated sequence
- Unit tests (5+ tests)
- Integration tests (2 tests)
- HTTP API response extension (stop_reason)

**Dependencies**:
- ✅ History buffer (Phase 2)
- ✅ Tokenizer integration (existing)

---

### Phase 4: Min-P and HTTP API (Day 7)
**Stories**: FT-019-EXT-4 (Min-P) + FT-019-EXT-5 (HTTP API)  
**Duration**: 0.5 + 0.5 = 1 day  
**Priority**: LOW (Min-P), HIGH (HTTP API)

**Why Last**:
- Min-P is optional (low priority, rarely used)
- HTTP API extension requires all kernels to be complete
- Validation logic needs all parameter ranges defined
- Backward compatibility testing needs full pipeline

**Deliverables (Min-P)**:
- `apply_min_p()` kernel
- Unit tests (3+ tests)

**Deliverables (HTTP API)**:
- Extended request schema (5 new parameters)
- Validation logic for all parameters
- Extended response schema (stop_reason)
- Error types and messages
- Unit tests (5+ tests: validation)
- Integration tests (3+ tests: full pipeline)
- Backward compatibility verification

**Dependencies**:
- ✅ All filtering kernels (Phases 1-3)
- ✅ Stop sequences (Phase 3)

---

## Summary: Correct Execution Order

```
Day 1-3:  FT-019-EXT-1 (Top-K + Top-P)
          ↓
Day 4:    FT-019-EXT-2 (Repetition Penalty)
          ↓
Day 5-6:  FT-019-EXT-3 (Stop Sequences)
          ↓
Day 7:    FT-019-EXT-4 (Min-P) + FT-019-EXT-5 (HTTP API)
```

---

## Dependency Graph

```
Sprint 3 (Complete)
├── Temperature Scaling (FT-017)
├── Greedy Sampling (FT-018)
└── Stochastic Sampling Core (FT-019)
    │
    ├─→ FT-019-EXT-1: Top-K + Top-P (Days 1-3)
    │   └─→ Independent, no dependencies
    │
    ├─→ FT-019-EXT-2: Repetition Penalty (Day 4)
    │   └─→ Creates history buffer infrastructure
    │
    ├─→ FT-019-EXT-3: Stop Sequences (Days 5-6)
    │   └─→ Depends on: History buffer (from EXT-2)
    │
    └─→ FT-019-EXT-4: Min-P (Day 7)
        └─→ FT-019-EXT-5: HTTP API (Day 7)
            └─→ Depends on: All kernels (EXT-1, EXT-2, EXT-3, EXT-4)
```

---

## Critical Path

**Longest dependency chain**: 7 days (sequential)

1. **Top-K/Top-P** (3 days) - Can start immediately
2. **Repetition Penalty** (1 day) - Can start after Top-K/Top-P OR in parallel
3. **Stop Sequences** (2 days) - MUST wait for Repetition Penalty (history buffer)
4. **Min-P + HTTP API** (1 day) - MUST wait for all kernels

**Optimization Opportunity**: 
- Top-K/Top-P (Days 1-3) and Repetition Penalty (Day 4) are independent
- Could parallelize if multiple agents available
- However, Foundation-Alpha works sequentially (per personality)

---

## Rationale for This Order

### Why Top-K/Top-P First?
1. **Most commonly used**: OpenAI, llama.cpp, LM Studio all have these
2. **Independent**: No dependencies on other advanced parameters
3. **Foundation**: Other parameters can be tested with these in place
4. **Complexity**: Medium complexity, good warm-up for Sprint 4

### Why Repetition Penalty Second?
1. **History infrastructure**: Creates history buffer needed by stop sequences
2. **Independent of filters**: Doesn't depend on top-k/top-p
3. **Commonly used**: Competitive parity with llama.cpp
4. **Moderate complexity**: 1 day is achievable

### Why Stop Sequences Third?
1. **Depends on history**: Needs history buffer from repetition penalty
2. **Critical feature**: Required for structured output (JSON, code)
3. **Complex**: Pattern matching, tokenization, early termination
4. **User-facing**: Affects response schema (stop_reason)

### Why Min-P + HTTP API Last?
1. **Min-P is optional**: Low priority, rarely used (can skip if time-constrained)
2. **HTTP API needs all kernels**: Must wait for complete pipeline
3. **Validation requires ranges**: All parameters must be defined
4. **Backward compatibility**: Needs full pipeline to test

---

## Testing Strategy

### Unit Tests (Per Story)
- **FT-019-EXT-1**: 10 tests (5 top-k, 5 top-p)
- **FT-019-EXT-2**: 4 tests (repetition penalty)
- **FT-019-EXT-3**: 5 tests (stop sequences)
- **FT-019-EXT-4**: 3 tests (min-p)
- **FT-019-EXT-5**: 5 tests (HTTP validation)

**Total**: 27 unit tests

### Integration Tests (Per Story)
- **FT-019-EXT-1**: 3 tests (combined top-k/top-p)
- **FT-019-EXT-2**: 2 tests (with temperature, with filters)
- **FT-019-EXT-3**: 2 tests (generation stops on match)
- **FT-019-EXT-5**: 3 tests (full pipeline, stop reasons)

**Total**: 10 integration tests

### End-to-End Test (Sprint 4 Complete)
```bash
curl -X POST http://localhost:8080/execute \
  -H "Content-Type: application/json" \
  -d '{
    "job_id": "test-advanced-sampling",
    "prompt": "Write a haiku about GPU computing",
    "max_tokens": 100,
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "min_p": 0.05,
    "stop": ["\n\n", "END"],
    "seed": 42
  }'
```

**Expected**: Generation with all parameters applied, stops on "\n\n" or "END"

---

## Performance Budget

| Parameter | Target Latency | Memory Overhead |
|-----------|----------------|-----------------|
| Top-K | <2ms | <1 MB (sorting) |
| Top-P | <1ms | <1 MB (sorting) |
| Repetition Penalty | <0.5ms | ~4 KB (history) |
| Stop Sequences | <0.1ms | ~512 bytes |
| Min-P | <0.1ms | 0 bytes |
| **Total** | **<5ms per token** | **<2 MB** |

---

## Backward Compatibility

All new parameters are **optional** with **sensible defaults**:

```json
{
  "temperature": 1.0,      // No scaling (identity)
  "top_p": 1.0,            // Disabled (no filtering)
  "top_k": 0,              // Disabled (no filtering)
  "repetition_penalty": 1.0, // Disabled (no penalty)
  "min_p": 0.0,            // Disabled (no filtering)
  "stop": []               // No stop sequences
}
```

**Old requests (Sprint 3 format) continue to work**:
```json
{
  "job_id": "job-xyz",
  "prompt": "Write a haiku",
  "max_tokens": 100,
  "temperature": 0.7,
  "seed": 42
}
```

---

## Definition of Done (Sprint 4)

- [ ] All 5 stories complete (FT-019-EXT-1 through FT-019-EXT-5)
- [ ] 27 unit tests passing
- [ ] 10 integration tests passing
- [ ] End-to-end test passing (all parameters)
- [ ] Performance within budget (<5ms per token)
- [ ] Backward compatibility verified
- [ ] HTTP API documentation updated
- [ ] Spec updated (M0-W-1421, M0-W-1422, M0-W-1300)
- [ ] Sprint 4 retrospective complete

---

## References

- **Primary Spec**: `bin/.specs/01_M0_worker_orcd.md` (M0-W-1421, M0-W-1422, M0-W-1300)
- **Analysis**: `bin/.specs/.docs/GENERATION_PARAMETERS_ANALYSIS.md`
- **Sprint 3 Complete**: `../sprint-3-shared-kernels/completed/FT-019-stochastic-sampling.md`
- **Story Files**: `./todo/FT-019-EXT-*.md`

---

## Next Steps for Foundation-Alpha

1. **Read this document** to understand execution order
2. **Start with FT-019-EXT-1** (Top-K + Top-P) - Days 1-3
3. **Move to FT-019-EXT-2** (Repetition Penalty) - Day 4
4. **Continue with FT-019-EXT-3** (Stop Sequences) - Days 5-6
5. **Finish with FT-019-EXT-4 + EXT-5** (Min-P + HTTP API) - Day 7

**Do NOT start out of order**. The dependencies are real and will cause rework.

---
Built by Foundation-Alpha 🏗️
