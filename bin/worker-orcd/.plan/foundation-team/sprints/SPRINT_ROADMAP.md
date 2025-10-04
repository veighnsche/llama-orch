# Foundation Team Sprint Roadmap

**Team**: Foundation-Alpha  
**Updated**: 2025-10-04  
**Status**: Active Planning

---

## Sprint Overview

| Sprint | Focus | Duration | Status | Stories |
|--------|-------|----------|--------|---------|
| Sprint 1 | HTTP Foundation | 5 days | ✅ Complete | HTTP server, SSE streaming |
| Sprint 2 | FFI Layer | 7 days | ✅ Complete | CUDA FFI, error handling |
| Sprint 3 | Shared Kernels | 7 days | ✅ Complete | Sampling, embedding, cuBLAS |
| Sprint 4 | Advanced Sampling | 7 days | 📋 Planned | Top-K, Top-P, repetition, stop |
| Sprint 5 | FP16 & Polish | 3-5 days | 📋 Planned | FP16, profiling, docs |
| Sprint 6 | Advanced Optimization | 2-3 days | 📋 Optional | CDF optimization, fused kernels |
| Sprint 7 | Final Integration | TBD | 📋 Planned | End-to-end M0 validation |

**Total Duration**: 31-39 days (6-8 weeks)

---

## Sprint 3: Shared Kernels ✅ COMPLETE

**Duration**: 7 days (Days 30-37)  
**Status**: ✅ Complete (2025-10-04)

### Completed Stories
- ✅ FT-016: cuBLAS wrapper (Day 30)
- ✅ FT-017: Temperature scaling (Day 32)
- ✅ FT-018: Greedy sampling (Day 33)
- ✅ FT-019: Stochastic sampling (Day 34-36)
- ✅ FT-020: Seeded RNG (Day 37)

### Deliverables
- Complete sampling pipeline (temperature → greedy/stochastic → RNG)
- 50+ unit tests
- Comprehensive documentation
- Foundation for production inference

### Deferred Work
- Advanced sampling parameters (top-p, top-k, etc.) → Sprint 4
- FP16 support → Sprint 5
- CDF optimization → Sprint 6

---

## Sprint 4: Advanced Sampling Parameters 📋 PLANNED

**Duration**: 7 days (Post-M0)  
**Status**: 📋 Planned  
**Priority**: High

### Stories
1. **FT-019-EXT-1**: Top-K and Top-P Sampling (3 days)
   - Top-K filtering with sorting
   - Top-P (nucleus) filtering
   - Integration tests

2. **FT-019-EXT-2**: Repetition Penalty (1 day)
   - History tracking
   - Penalty application
   - Integration with generation loop

3. **FT-019-EXT-3**: Stop Sequences (2 days)
   - Tokenization of stop strings
   - Pattern matching
   - Early termination

4. **FT-019-EXT-4**: Min-P Sampling (0.5 days, optional)
   - Minimum probability filtering
   - Unit tests

5. **FT-019-EXT-5**: HTTP API Extension (0.5 days)
   - Request schema extension
   - Validation logic
   - Backward compatibility

### Deliverables
- 5 new sampling kernels
- 25+ unit tests
- 5+ integration tests
- HTTP API extension
- Complete parameter documentation

### Success Criteria
- 8/10 parameters vs OpenAI (80% parity)
- All critical parameters implemented
- Backward compatibility maintained
- Performance within budget (<5ms per token)

---

## Sprint 5: FP16 Optimization & Polish 📋 PLANNED

**Duration**: 3-5 days (Post-Sprint 4)  
**Status**: 📋 Planned  
**Priority**: Medium

### Stories
1. **FT-021**: FP16 Sampling Support (1 day)
   - FP16 greedy sampling
   - FP16 stochastic sampling
   - FP16 advanced sampling
   - Performance benchmarks

2. **FT-023**: Sampling Performance Profiling (1 day)
   - Benchmark all kernels
   - Profile memory bandwidth
   - Identify bottlenecks
   - Optimization recommendations

3. **FT-025**: Documentation & Examples (1 day)
   - User-facing API docs
   - Example requests
   - Parameter interaction guide
   - Troubleshooting guide

### Deliverables
- FP16 sampling kernels
- Performance benchmark report
- User documentation
- Example requests

### Success Criteria
- FP16 performance improvement (20-30% faster)
- Complete user documentation
- All examples tested

---

## Sprint 6: Advanced Optimization 📋 OPTIONAL

**Duration**: 2-3 days (Post-Sprint 5)  
**Status**: 📋 Optional  
**Priority**: Low

### Stories
1. **FT-022**: Optimized CDF Computation (2 days)
   - Parallel prefix sum
   - Binary search sampling
   - Performance benchmarks

2. **FT-026**: Fused Sampling Kernels (1 day, optional)
   - Fuse temperature + softmax
   - Fuse filtering + softmax
   - Reduce kernel launches

### Deliverables
- Optimized CDF computation
- Fused kernels (optional)
- Performance comparison

### Decision Criteria
**Execute if**:
- Sampling latency > 5ms per token
- Memory bandwidth is bottleneck
- User feedback requests faster sampling

**Skip if**:
- Sampling latency < 5ms per token
- Other bottlenecks more critical
- M1 priorities take precedence

---

## Deferred Work Summary

### From Sprint 3
**Total**: 5-7 days of work deferred to Sprint 4

**Features**:
- Top-P (nucleus) sampling: 1-2 days
- Top-K sampling: 1 day
- Repetition penalty: 1 day
- Stop sequences: 1-2 days
- Min-P sampling: 0.5 days
- HTTP API extension: 0.5 days

**Rationale**: Core sampling sufficient for M0, advanced features deserve focused implementation

### From Sprint 4 (Potential)
**Total**: 2-3 days of work may defer to Sprint 5

**Features**:
- Min-P sampling (if time-constrained)
- Performance optimizations (if not critical)

---

## Timeline Projection

### Conservative Estimate (All Sprints)
- Sprint 4: 7 days
- Sprint 5: 5 days
- Sprint 6: 3 days
- **Total**: 15 days (3 weeks)

### Optimistic Estimate (Skip Sprint 6)
- Sprint 4: 6 days (defer Min-P)
- Sprint 5: 3 days (minimal polish)
- Sprint 6: Skipped
- **Total**: 9 days (2 weeks)

### Realistic Estimate
- Sprint 4: 7 days (full implementation)
- Sprint 5: 4 days (FP16 + docs)
- Sprint 6: Conditional (0-3 days)
- **Total**: 11-14 days (2-3 weeks)

---

## Risk Management

### Sprint 4 Risks
**High**: Sorting performance may be unacceptable
- **Mitigation**: Use Thrust, profile early, optimize if needed

**Medium**: Parameter interactions may have bugs
- **Mitigation**: Comprehensive integration tests

**Low**: Scope creep
- **Mitigation**: Strict scope definition, defer Min-P if needed

### Sprint 5 Risks
**Low**: FP16 accuracy issues
- **Mitigation**: Thorough testing, compare vs FP32

**Low**: Documentation incomplete
- **Mitigation**: Allocate full day for docs

### Sprint 6 Risks
**Low**: Optimization doesn't improve performance
- **Mitigation**: Profile before implementing, skip if not needed

---

## Success Metrics

### Sprint 4 Success
- ✅ All advanced parameters implemented
- ✅ 80% parameter parity with OpenAI
- ✅ Backward compatibility maintained
- ✅ Performance within budget

### Sprint 5 Success
- ✅ FP16 support complete
- ✅ 20-30% performance improvement
- ✅ Complete user documentation

### Sprint 6 Success (If Executed)
- ✅ Additional performance improvement
- ✅ Sampling latency < 3ms per token

---

## Post-Sprint Plan

### After Sprint 4
**Decision Point**: Evaluate M0 readiness
- If M0 ready: Proceed to M0 validation
- If polish needed: Execute Sprint 5
- If performance critical: Execute Sprint 6

### After Sprint 5
**Decision Point**: Evaluate performance
- If performance acceptable: Skip Sprint 6, proceed to M0 validation
- If performance critical: Execute Sprint 6

### After Sprint 6
**Proceed to**: M0 validation and integration testing

---

## References

- **Deferred Work Backlog**: `DEFERRED_WORK_BACKLOG.md`
- **Sprint 3 Summary**: `sprint-3-shared-kernels/SPRINT_SUMMARY.md`
- **Sprint 4 Plan**: `sprint-4-advanced-sampling/README.md`
- **Sprint 5 Plan**: `sprint-5-fp16-optimization/README.md`
- **Sprint 6 Plan**: `sprint-6-advanced-optimization/README.md`

---
Built by Foundation-Alpha 🏗️
