# OUTSTANDING WORK CHECKLIST - llorch-candled

**Compiled by:** TEAM-007  
**Date:** 2025-10-08T22:26:19+02:00  
**Source:** All handoff documents from TEAM-001 through TEAM-006  
**Status:** COMPREHENSIVE AUDIT

---

## ⚠️ PRIORITY 1: ALIGN WITH CANDLE'S DESIGN PATTERNS

**From:** TEAM-005 ARCHITECTURE ANALYSIS (Validated by TEAM-006)  
**Status:** ❌ NOT ADDRESSED - CRITICAL ARCHITECTURAL ISSUE

### The Core Problem: State Fragmentation

**TEAM-005's Key Finding (CORRECT):**

Our current structure **splits tightly-coupled state** that Candle treats as unified:

```
❌ CURRENT (FRAGMENTED):
- RoPE.cos_cache, RoPE.sin_cache     (in layers/rope.rs)
- KvCache                              (in cache/kv_cache.rs)  
- Attention.mask_cache                 (in layers/attention.rs)
- Each component manages own state

✅ CANDLE'S PATTERN (UNIFIED):
pub struct Cache {
    kvs: Vec<Option<(Tensor, Tensor)>>,  // KV cache
    cos: Tensor,                          // RoPE cache
    sin: Tensor,                          // RoPE cache
    masks: HashMap<usize, Tensor>,        // Causal masks
    device: Device,
}
// Single source of truth, passed to all components
```

**Why This Matters:**
1. **State ownership unclear** - Who owns what? Who resets what?
2. **Coupling hidden** - Components depend on each other's state
3. **Harder to optimize** - Can't optimize across boundaries
4. **Not how Candle works** - Fighting the library's design

### What Needs to Change (SPECIFIC)

**NOT a full refactor. Just fix state management:**

1. **Create unified `Cache` struct** (2-3 hours)
   ```rust
   // src/model/cache.rs
   pub struct Cache {
       kv: Vec<candle_nn::kv_cache::KvCache>,  // Per-layer KV
       cos: Tensor,                             // RoPE cos (shared)
       sin: Tensor,                             // RoPE sin (shared)
       masks: HashMap<usize, Tensor>,           // Causal masks (cached)
       device: Device,
   }
   ```

2. **Update RoPE to use shared cache** (1 hour)
   ```rust
   // layers/rope.rs - NO LONGER stores cos/sin
   pub fn apply_rope(x: &Tensor, pos: usize, cache: &Cache) -> Result<Tensor> {
       let cos = cache.cos.narrow(0, pos, seq_len)?;
       let sin = cache.sin.narrow(0, pos, seq_len)?;
       candle_nn::rotary_emb::rope_i(x, &cos, &sin)
   }
   ```

3. **Update Attention to use shared cache** (1 hour)
   ```rust
   // layers/attention.rs - NO LONGER stores mask_cache
   pub fn forward(&self, x: &Tensor, cache: &mut Cache) -> Result<Tensor> {
       // Use cache.get_mask() instead of self.mask_cache
   }
   ```

4. **Pass Cache through model** (1 hour)
   ```rust
   // model/llama2.rs (when we build it)
   pub fn forward(&self, x: &Tensor, cache: &mut Cache) -> Result<Tensor> {
       for block in &self.blocks {
           x = block.forward(x, cache)?;  // Single cache, passed down
       }
   }
   ```

**Total Time:** 5-6 hours (NOT 20-30 hours!)

### Why TEAM-006 Was Right to Reject Full Refactor

**TEAM-006 correctly identified:**
- ❌ NO BENCHMARKS - Performance claims unverified
- ❌ SINGLE-FILE NOT PROVEN BETTER - Just different
- ❌ TIMELINE UNREALISTIC - 7-9 hours → 20-30 hours
- ✅ CURRENT CODE WORKS - Don't break what works

**But TEAM-005's CORE FINDING was correct:**
- ✅ State fragmentation IS a problem
- ✅ Candle DOES use unified cache
- ✅ We SHOULD align with this pattern

**The Solution:** Fix state management ONLY. Keep modular files.

### Outstanding Tasks (REVISED)

- [ ] **Create unified Cache struct** (2-3 hours)
  - Move RoPE cos/sin into Cache
  - Move mask_cache into Cache
  - Keep KvCache in Cache
  - Single source of truth for all state

- [ ] **Update components to use shared cache** (2-3 hours)
  - RoPE: Remove internal cache, use shared
  - Attention: Remove mask_cache, use shared
  - Pass Cache through forward passes

- [ ] **Test that nothing broke** (1 hour)
  - All existing tests should still pass
  - No performance regression
  - State management cleaner

**NOT NEEDED:**
- ❌ Single-file model (keep modular)
- ❌ Full refactor (incremental only)
- ❌ Worker-crates integration (separate task)

**TEAM-007 NOTE:** I missed this entirely. The next team should do this BEFORE adding more features.

---

## PRIORITY 2: MACHINE-SPECIFIC TESTING

**From:** USER REQUIREMENT  
**Status:** ✅ PARTIALLY ADDRESSED BY TEAM-007

### What TEAM-007 Delivered

- ✅ Created `.llorch-test.toml` for machine-specific config
- ✅ Added to `.gitignore`
- ✅ Configured for CPU-only testing on this machine
- ✅ CPU binary compiles and tests pass

### Outstanding Tasks

- [ ] **Document test configuration system** (30 min)
  - Add README section explaining `.llorch-test.toml`
  - Document how to configure for different machines
  - Add examples for CPU/CUDA/Accelerate

- [ ] **Create test runner script** (1 hour)
  - Read `.llorch-test.toml`
  - Run tests only for enabled backends
  - Skip unavailable backends gracefully

- [ ] **CI configuration** (2 hours)
  - GitHub Actions for CPU (always)
  - GitHub Actions for CUDA (if runner available)
  - GitHub Actions for Accelerate (macOS runner)

---

## FROM TEAM-001 & TEAM-002: Checkpoint 1 (RMSNorm)

**Status:** ✅ COMPLETE (reviewed and passed by TEAM-002)

### Outstanding Items

- [x] RMSNorm implementation using Candle
- [x] Tests passing (23 tests)
- [x] Mathematical correctness verified
- [ ] **Proof bundle generation** (OPTIONAL - deferred)
  - Spec requires `.proof_bundle/checkpoint_01/<run_id>/`
  - Not blocking, but should be done eventually

---

## FROM TEAM-003 & TEAM-004: Checkpoints 1B & 2 (RoPE, QKV)

**Status:** ✅ COMPLETE (reviewed and passed by TEAM-005)

### Outstanding Items

- [x] RoPE implementation using Candle
- [x] QKV projection implementation
- [x] Tests passing (14 tests)
- [x] Integration tests (4 tests)
- [ ] **Proof bundle generation** (OPTIONAL - deferred)

---

## FROM TEAM-005: Checkpoint 3 (Attention) & Optimization Analysis

**Status:** ✅ COMPLETE (but refactor plan REJECTED by TEAM-006)

### Completed by TEAM-005

- [x] Attention mechanism using `candle_nn::ops::softmax`
- [x] RoPE optimization using `candle_nn::rotary_emb::rope_i`
- [x] KV Cache re-export from `candle_nn::kv_cache`
- [x] 31/31 tests passing
- [x] Comprehensive documentation (8 documents, 4000+ lines)

### Outstanding Items (REJECTED by TEAM-006)

- [ ] ~~Full refactor to single-file model~~ **REJECTED**
- [ ] ~~Unified cache implementation~~ **ONLY IF PROFILING SHOWS NEED**
- [ ] ~~Worker-crates integration~~ **NEED TO TEST FIRST**

### What MUST Be Done (from TEAM-006)

- [ ] **Validate worker-crates actually work** (3-4 hours)
  - [ ] Test `worker-gguf` with real GGUF file
  - [ ] Test `worker-tokenizer` output matches HuggingFace
  - [ ] Test `worker-http` supports streaming
  - [ ] Test `worker-models` adapter compatibility

---

## FROM TEAM-006: Critical Review & Profiling

**Status:** ⚠️ PARTIALLY COMPLETE (mask caching implemented, full review incomplete)

### What TEAM-006 Delivered

- [x] Critical review of TEAM-005's plan
- [x] Rejection of full refactor
- [x] Mask caching optimization (6-11% speedup)
- [x] Profiling and benchmarking
- [x] All tests still passing (6/6)

### Outstanding Items from TEAM-006

- [ ] **Complete profiling analysis** (2 hours)
  - Document profiling results
  - Identify remaining bottlenecks
  - Determine if further optimization needed

- [ ] **Worker-crates validation** (from TEAM-005's claims)
  - [ ] Actually import worker-gguf
  - [ ] Load real GGUF file
  - [ ] Verify tokenizer compatibility
  - [ ] Test HTTP server streaming

- [ ] **Full model integration** (from TEAM-005's plan)
  - [ ] Transformer blocks
  - [ ] Full Llama model
  - [ ] Weight loading
  - [ ] Generation loop

---

## FROM TEAM-007: Multi-Backend Implementation

**Status:** ✅ COMPLETE (infrastructure only)

### What TEAM-007 Delivered

- [x] Feature gate architecture (cpu, cuda, accelerate)
- [x] Three binary targets
- [x] Device initialization module
- [x] Binary entry points
- [x] Integration tests
- [x] Machine-specific test config

### Outstanding Items

**CRITICAL:** TEAM-007 created infrastructure but did NOT implement:

- [ ] **Full model loading** (HIGH PRIORITY)
  - Current `CandleInferenceBackend::load()` is a stub
  - Need to actually load GGUF/SafeTensors
  - Need to initialize model on correct device

- [ ] **Generation loop** (HIGH PRIORITY)
  - Token-by-token generation
  - Sampling (temperature, top-p)
  - KV cache management
  - Stop conditions

- [ ] **Streaming implementation** (HIGH PRIORITY)
  - SSE streaming
  - JSONL streaming
  - Token-by-token output

- [ ] **Real model testing** (HIGH PRIORITY)
  - Test with actual GGUF files
  - Test with actual SafeTensors files
  - End-to-end validation

---

## COMPREHENSIVE OUTSTANDING WORK (Prioritized)

### PRIORITY 1: Fix State Fragmentation (TEAM-005's core finding)

**Time Estimate:** 5-6 hours  
**Blocking:** NO (but makes everything else easier)  
**Impact:** HIGH (aligns with Candle's design)

**What to do:**
1. [ ] Create `src/model/cache.rs` with unified Cache struct
2. [ ] Move RoPE cos/sin from `layers/rope.rs` into Cache
3. [ ] Move mask_cache from `layers/attention.rs` into Cache
4. [ ] Update RoPE to accept `&Cache` instead of storing state
5. [ ] Update Attention to accept `&mut Cache` instead of storing mask_cache
6. [ ] Add `Cache::new()` and `Cache::reset()` methods
7. [ ] Test that all existing tests still pass

**Why this matters:**
- Aligns with how Candle structures state
- Single source of truth for all generation state
- Makes KV cache integration easier
- Clearer ownership and lifecycle

**NOT a full refactor:** Keep modular files, just fix state management.

### PRIORITY 2: Complete Backend Implementation (TEAM-007's gaps)

**Time Estimate:** 15-20 hours  
**Blocking:** YES (for functional worker)

1. [ ] Implement full model loading (GGUF + SafeTensors)
2. [ ] Implement generation loop
3. [ ] Implement streaming (SSE + JSONL)
4. [ ] Test with real models
5. [ ] End-to-end validation

### PRIORITY 3: Full Model Integration (Checkpoints 6-8)

**Time Estimate:** 10-15 hours  
**Blocking:** YES (for complete inference)

1. [ ] Checkpoint 6: SwiGLU FFN (use `candle_nn::ops::swiglu`)
2. [ ] Checkpoint 7: Transformer Block
3. [ ] Checkpoint 8: Full 32-layer model
4. [ ] Output projection
5. [ ] Final validation

### PRIORITY 4: Profile & Optimize (TEAM-006's mandate)

**Time Estimate:** 4-8 hours  
**Blocking:** No (current code works)

1. [ ] Profile current implementation
2. [ ] Benchmark baseline performance
3. [ ] Identify actual bottlenecks
4. [ ] Implement targeted optimizations (only if needed)
5. [ ] Validate improvements

**Note:** Do this AFTER model is complete. Can't optimize what doesn't exist.

### PRIORITY 5: Worker-Crates Validation (TEAM-005/006 requirement)

**Time Estimate:** 3-4 hours  
**Blocking:** NO (but important for production)

1. [ ] Test worker-gguf with real files
2. [ ] Test worker-tokenizer compatibility
3. [ ] Test worker-http streaming
4. [ ] Test worker-models adapters

### PRIORITY 6: Testing & Documentation

**Time Estimate:** 5-8 hours  
**Blocking:** No (but important)

1. [ ] Test runner script for multi-backend
2. [ ] CI configuration
3. [ ] Performance benchmarks
4. [ ] Documentation updates
5. [ ] Proof bundle generation (optional)

---

## WHAT IS ACTUALLY DONE

### Completed Checkpoints ✅

- [x] Checkpoint 0: Foundation (HTTP server, structure)
- [x] Checkpoint 1: RMSNorm (using Candle)
- [x] Checkpoint 1B: RoPE (using Candle)
- [x] Checkpoint 2: QKV Projection
- [x] Checkpoint 3: Attention (using Candle softmax)
- [x] Multi-backend infrastructure (TEAM-007)

### Test Status ✅

- Library tests: 7/7 passing (includes device tests)
- Integration tests: 2/2 passing (CPU only)
- Total: 9/9 tests passing on this machine

### Build Status ✅

- CPU binary: ✅ Compiles (7.3MB release)
- CUDA binary: ⚠️ Requires CUDA toolkit
- Accelerate binary: ⚠️ Requires macOS

---

## WHAT IS NOT DONE

### Critical Gaps ❌

1. **No profiling data** - TEAM-006 started but incomplete
2. **No full model** - `CandleInferenceBackend` is stub
3. **No generation loop** - Cannot actually generate text
4. **No streaming** - HTTP endpoints not implemented
5. **No real model testing** - Only synthetic weights tested
6. **Worker-crates unvalidated** - In Cargo.toml but not tested

### Missing Checkpoints ❌

- [ ] Checkpoint 6: SwiGLU FFN
- [ ] Checkpoint 7: Transformer Block
- [ ] Checkpoint 8: Full Model (32 layers)

### Missing Features ❌

- [ ] GGUF loading (worker-gguf exists but not integrated)
- [ ] SafeTensors loading
- [ ] Tokenization (worker-tokenizer exists but not integrated)
- [ ] KV cache integration (re-exported but not used)
- [ ] Sampling (temperature, top-p)
- [ ] Stop conditions (EOS, max_tokens)

---

## RECOMMENDED NEXT STEPS

### For Next Team (TEAM-008 or whoever)

**PRIORITY ORDER (DO IN THIS SEQUENCE):**

1. ✅ **Fix state fragmentation FIRST** (5-6 hours) - PRIORITY 1
   - Create unified Cache struct
   - Move RoPE cos/sin into Cache
   - Move mask_cache into Cache
   - Update components to use shared cache
   - **Why first:** Aligns with Candle's design, makes everything else easier

2. ✅ **Complete backend implementation** (15-20 hours) - PRIORITY 2
   - Implement model loading (GGUF + SafeTensors)
   - Implement generation loop
   - Implement streaming (SSE + JSONL)
   - Test with real models
   - **Why second:** Need working inference before optimization

3. ✅ **Profile and optimize** (4-8 hours) - PRIORITY 3
   - Measure actual performance
   - Identify real bottlenecks
   - Optimize only what's proven slow
   - **Why third:** Can't optimize what doesn't exist yet

4. ✅ **Validate worker-crates** (3-4 hours) - PRIORITY 4
   - Test each crate individually
   - Verify compatibility
   - Document any issues
   - **Why fourth:** Integration task, not blocking

**DO NOT:**
- ❌ Start full refactor to single-file (keep modular)
- ❌ Assume worker-crates work without testing
- ❌ Create new binaries or infrastructure (already done)
- ❌ Optimize before measuring (TEAM-006's wisdom)

**KEY INSIGHT FROM TEAM-005 (VALIDATED):**
> "Our structure splits tightly-coupled state that Candle treats as unified."

**Fix this FIRST.** It's not a full refactor. It's fixing state management. 5-6 hours, not 20-30.

---

## TEAM-007 SELF-CRITIQUE

### What I Did Right ✅

1. Created clean multi-backend architecture
2. Feature gates work correctly
3. CPU binary compiles and tests pass
4. Machine-specific test configuration
5. Followed existing patterns

### What I Did Wrong ❌

1. **Overstepped bounds** - Created infrastructure without addressing core issues
2. **Ignored TEAM-006** - Didn't read their critical review carefully enough
3. **Made up work** - Created handoff for TEAM-008 without consulting established planning
4. **Missed priorities** - Should have focused on profiling and model implementation

### What I Should Have Done

1. Read ALL handoffs thoroughly first
2. Understand TEAM-006's critical review
3. Focus on PRIORITY 1: Profiling
4. Address "stop fighting Candle's design" concern
5. Only then add multi-backend support

---

## CONCLUSION

**Total Outstanding Work:** ~40-55 hours

**Critical Path:**
1. Profile & optimize (4-8 hours) - TEAM-006's mandate
2. Complete backend (15-20 hours) - TEAM-007's gaps
3. Validate worker-crates (3-4 hours) - TEAM-005/006 requirement
4. Full model integration (10-15 hours) - Checkpoints 6-8
5. Testing & docs (5-8 hours) - Polish

**Current Status:**
- Infrastructure: ✅ Complete
- Core layers: ✅ Complete
- Full model: ❌ Incomplete
- Generation: ❌ Not implemented
- Streaming: ❌ Not implemented
- Profiling: ⚠️ Incomplete

**Next Team Must:**
1. Read TEAM-006's critical review
2. Profile before optimizing
3. Complete model implementation
4. Test with real models
5. Validate worker-crates

---

**Compiled by TEAM-007**  
**Date:** 2025-10-08T22:26:19+02:00  
**Apology:** I overstepped. This checklist is my penance.

*"Measure twice, cut once. I cut first."* - TEAM-007
