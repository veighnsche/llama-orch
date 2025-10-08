# Executive Summary: llorch-candled Optimization

**Date:** 2025-10-08  
**Team:** TEAM-005  
**Status:** ANALYSIS COMPLETE - READY TO EXECUTE

---

## What We Discovered

### 1. We're Fighting Candle's Design ❌

**Problem:** Our architecture splits what Candle treats as tightly coupled:
- **Cache fragmentation:** RoPE cache separate from KV cache
- **Pipeline splitting:** QKV → RoPE → Attention (3 separate structs)
- **State scattered:** Multiple structs managing their own state
- **Mask recreation:** Causal masks created every forward pass

**Candle's pattern:**
- Single `Cache` struct with all state
- Integrated `CausalSelfAttention` (QKV + RoPE + Attention in one)
- Single-file model (everything in `llama.rs`)

### 2. We're Reinventing Infrastructure ❌

**Problem:** We were planning to implement:
- GGUF parsing
- Tokenization
- Model configuration
- HTTP server

**Reality:** All already exist in worker-crates!
- ✅ `worker-gguf` - Complete GGUF parser
- ✅ `worker-tokenizer` - Full BPE tokenizer
- ✅ `worker-models` - Model configs and detection
- ✅ `worker-http` - HTTP server with streaming

**We can reuse 90% of infrastructure!**

### 3. We're Using Candle Correctly ✅

**Good news:** We ARE using Candle's ops correctly:
- ✅ `candle_nn::rotary_emb::rope_i`
- ✅ `candle_nn::ops::rms_norm`
- ✅ `candle_nn::ops::softmax`
- ✅ `candle_nn::kv_cache::KvCache`

**Just need better architecture around them.**

---

## The Solution

### Align with Candle + Reuse Worker Crates

**Two-pronged approach:**

1. **Reuse Infrastructure** (worker-crates)
   - GGUF parsing → `worker-gguf`
   - Tokenization → `worker-tokenizer`
   - Config → `worker-models`
   - HTTP → `worker-http`

2. **Refactor to Candle Pattern** (single-file model)
   - Unified `Cache` (KV + RoPE + masks)
   - Integrated `CausalSelfAttention`
   - Single file: `model/llama2.rs`

---

## Expected Benefits

### Performance
- **2-3x faster** (Candle GPU kernels)
- **Better caching** (masks cached, not recreated)
- **Integrated pipeline** (no function call overhead)

### Code Quality
- **90% code reuse** (worker-crates)
- **~200 lines removed** (consolidation)
- **Simpler architecture** (single-file model)
- **Better state management** (centralized cache)

### Maintainability
- **Follows proven pattern** (Candle's design)
- **Less custom code** (more library usage)
- **Clearer ownership** (cache owns all state)

---

## What Changes

### Delete
```
src/layers/rope.rs          ❌ (move to model/llama2.rs)
src/layers/attention.rs     ❌ (move to model/llama2.rs)
src/layers/swiglu.rs        ❌ (move to model/llama2.rs)
src/layers/rms_norm.rs      ❌ (move to model/llama2.rs)
src/layers/transformer.rs   ❌ (move to model/llama2.rs)
src/tensor/                 ❌ (use Candle tensors)
src/backend/                ❌ (use Candle device)
```

### Create
```
src/model/cache.rs          ✅ Unified cache
src/model/llama2.rs         ✅ Single-file model (Candle pattern)
```

### Add Dependencies
```toml
worker-gguf       ✅ GGUF parsing
worker-tokenizer  ✅ Tokenization
worker-models     ✅ Config
worker-http       ✅ HTTP server
```

---

## Timeline

### Phase 1: Add Dependencies (30 min)
- Update `Cargo.toml`
- Verify build

### Phase 2: Unified Cache (1 hour)
- Create `src/model/cache.rs`
- Centralize KV + RoPE + masks

### Phase 3: Single-File Model (3-4 hours)
- Create `src/model/llama2.rs`
- Integrate: Cache, CausalSelfAttention, Mlp, Block, Llama
- Follow Candle's pattern

### Phase 4: Delete Old Code (30 min)
- Remove redundant layers/
- Clean up imports

### Phase 5: Update Main (1 hour)
- Use worker-crates
- Wire everything together

### Phase 6: Tests (2 hours)
- Update tests
- Verify functionality

### Phase 7: Documentation (1 hour)
- Update README
- Update specs

**Total: 7-9 hours**

---

## Risk Assessment

### Low Risk ✅
- Worker crates are battle-tested
- Candle pattern is proven
- Can rollback if needed

### Mitigation
- Keep old code in branch
- Incremental migration
- Test at each phase

---

## Documents Created

1. **CANDLE_USAGE_POLICY.md** - Policy for using Candle
2. **ARCHITECTURE_ANALYSIS.md** - Deep dive on structure issues
3. **WORKER_CRATES_ANALYSIS.md** - What we can reuse
4. **REFACTOR_PLAN.md** - Step-by-step execution plan
5. **EXECUTIVE_SUMMARY.md** - This document

---

## Recommendation

**Execute the refactor.** 

**Why:**
- Aligns with proven patterns
- Massive code reuse
- Better performance
- Cleaner architecture

**When:**
- Start with Phase 1 (dependencies)
- Execute incrementally
- Test at each phase

**Confidence:** HIGH ✅

---

## Key Takeaways

1. ✅ **Use Candle for ops** - We're doing this right
2. ❌ **Don't split architecture** - We need to fix this
3. ✅ **Reuse infrastructure** - Worker crates are ready
4. ✅ **Follow proven patterns** - Candle's design works

**Bottom line:** We're 80% there. Just need to align architecture and reuse infrastructure.

---

**Ready to execute!** 🚀

---

*Summary by TEAM-005, 2025-10-08*
