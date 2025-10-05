# Sprint 9 Stories V2: Best of Both Worlds

**Date**: 2025-10-05  
**Version**: 2.0 (Revised with llama.cpp + vLLM patterns)  
**Status**: PROPOSED

---

## What Changed from V1

### V1 Approach (Original)
- ❌ Hardcoded architecture-specific logic
- ❌ Simple contiguous KV cache
- ❌ Single-request inference only
- ✅ Fast to implement (22-31 hours)

### V2 Approach (Best of Both Worlds)
- ✅ Data-driven architecture registry (llama.cpp)
- ✅ Paged KV cache with block manager (vLLM)
- ✅ Batch-ready infrastructure (vLLM)
- ⚠️ More complex (28-38 hours, +6-7h)

---

## Story Comparison

| Story | V1 | V2 | Change |
|-------|----|----|--------|
| GT-051 | Config parsing (hardcoded) | ✅ DONE | Refactor in GT-058 |
| GT-052 | Weight loading (simple) | Weight loading + registry | +Registry |
| GT-053 | BPE tokenizer | BPE tokenizer | No change |
| GT-054 | Transformer (contiguous KV) | Transformer (paged KV) | +Paged cache |
| GT-055 | LM head | LM head | No change |
| GT-056 | Wire inference | Wire inference + blocks | +Block mgmt |
| GT-057 | Test cleanup | Test cleanup | No change |
| GT-058 | - | Refactor GT-051 | New story |

---

## V2 Stories

### GT-051: ✅ COMPLETE
**Status**: Done, but needs refactor (see GT-058)

### GT-052-V2: Architecture Registry + Weight Loading
**New components**:
1. `ArchitectureRegistry` - Data-driven arch configs
2. `MetadataKeyMapper` - Dynamic key construction
3. `TensorMapper` - Pattern expansion + feature probing
4. `GGUFWeightIterator` - Clean weight iteration

**Size**: L (was M)  
**Estimate**: 8-10 hours (was 6-8)  
**Why more**: Registry infrastructure is one-time investment

### GT-053-V2: BPE Tokenizer
**No changes from V1**

**Size**: M  
**Estimate**: 5-7 hours

### GT-054-V2: Paged KV Cache + Transformer
**New components**:
1. `PagedKVCache` - Block pool manager
2. `KVBlockAllocator` - Dynamic block allocation
3. Updated attention kernels - Block table support

**Size**: M (was M, but more complex)  
**Estimate**: 6-8 hours (was 4-6)  
**Why more**: Paged cache is more complex than contiguous

### GT-055-V2: LM Head
**No changes from V1**

**Size**: S  
**Estimate**: 2-3 hours

### GT-056-V2: Wire Inference with Block Management
**New components**:
1. Block allocation per request
2. Block deallocation on completion
3. Block table management

**Size**: S (was S, slightly more complex)  
**Estimate**: 3-4 hours (was 2-3)  
**Why more**: Block lifecycle management

### GT-057-V2: Test Cleanup
**No changes from V1**

**Size**: XS  
**Estimate**: 1-2 hours

### GT-058: NEW - Refactor GT-051 with MetadataKeyMapper
**New story**:
1. Extract `MetadataKeyMapper` class
2. Refactor `parse_config_from_gguf()` to use mapper
3. Remove hardcoded key construction
4. All tests still pass

**Size**: S  
**Estimate**: 2-3 hours  
**Priority**: Medium (can be done post-M0)

---

## Total Estimates

### V1 (Original)
- **Total**: 22-31 hours
- **Approach**: Fast, hardcoded, single-request
- **Risk**: Hard to add new models later

### V2 (Best of Both Worlds)
- **Total**: 28-38 hours
- **Approach**: Data-driven, paged cache, batch-ready
- **Risk**: More complex upfront

**Difference**: +6-7 hours (27% increase)

---

## Decision Matrix

### Option A: Ship V1 Now, Refactor Later
**Pros**:
- ✅ Faster to M0 (22-31 hours)
- ✅ Simpler implementation
- ✅ Gets haiku test passing quickly

**Cons**:
- ❌ Hardcoded architecture logic
- ❌ Adding Phi-3, GPT-OSS-20B requires code changes
- ❌ Contiguous KV cache is memory-inefficient
- ❌ Refactor later costs more (2-3 weeks)

### Option B: Implement V2 Now
**Pros**:
- ✅ Data-driven, no hardcoding
- ✅ Add new models by config, not code
- ✅ Paged KV cache is production-ready
- ✅ Batch-ready infrastructure
- ✅ One-time investment

**Cons**:
- ❌ +6-7 hours to M0
- ❌ More complex to implement
- ❌ More complex to debug

### Option C: Hybrid (Recommended)
**Approach**:
1. Ship GT-051 (done), GT-052-V1, GT-053-V1 (simple versions)
2. Get haiku test passing
3. Then refactor to V2 patterns (GT-058, GT-052-V2, GT-054-V2)

**Pros**:
- ✅ Fast to M0 (use V1)
- ✅ Proven working before refactor
- ✅ Can validate V2 patterns incrementally

**Cons**:
- ❌ Some throwaway code
- ❌ Total time: V1 + refactor = ~35-45 hours

---

## Recommendation

**Implement V2 directly** for these reasons:

1. **One-time cost**: 6-7 extra hours now vs 2-3 weeks later
2. **M0 deadline**: Still achievable (28-38 hours = 4-5 days)
3. **Quality**: Production-ready patterns from day 1
4. **Maintainability**: Clear architecture, easy to extend
5. **No throwaway code**: Everything we build is final

**Alternative** (if deadline is critical):
- Implement V1, mark as "refactor needed"
- Add GT-059 to GT-065 for V2 refactor
- Ship M0 faster, refactor in M1

---

## Next Steps

**If V2 approved**:
1. ⬜ Create detailed GT-052-V2 story
2. ⬜ Create detailed GT-054-V2 story
3. ⬜ Create detailed GT-056-V2 story
4. ⬜ Create detailed GT-058 story
5. ⬜ Update sprint timeline

**If V1 preferred**:
1. ⬜ Keep existing stories
2. ⬜ Add "refactor needed" tags
3. ⬜ Create M1 refactor stories

---

**Created by**: Project Management Team 📋  
**Date**: 2025-10-05  
**Decision needed**: V1 (fast) vs V2 (quality) vs Hybrid?

---
Reviewed by Testing Team 🔍
