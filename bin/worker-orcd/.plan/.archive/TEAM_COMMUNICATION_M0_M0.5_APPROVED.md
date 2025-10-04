# Team Communication: M0 + M0.5 Approved

**Date**: 2025-10-04 02:42  
**From**: Project Manager (M0 Worker-orcd)  
**To**: All Teams (Foundation-Alpha, Llama-Beta, GPT-Gamma, Performance Team)  
**Subject**: M0 Day 1 Launch - PROCEED with Clean Seams for M0.5

---

## TL;DR

✅ **M0 Day 1 Launch: PROCEED** (110 days, 139 stories, unchanged)  
✅ **M0.5 Approved**: Post-M0 performance foundation (+2 weeks)  
✅ **Clean Seams Required**: Structure M0 for easy M0.5/M1 integration

---

## Decision Summary

**Management has approved the M0 + M0.5 plan**:

### M0 (Days 1-110) - UNCHANGED ✅
- 3 models: Qwen, Phi-3, GPT-OSS-20B
- 3 quantization formats: Q4_K_M, MXFP4, Q4_0
- Architecture adapters (clean design)
- Critical safety (VRAM monitoring, OOM handling)
- 139 story cards ready
- **Day 1 launch: PROCEED**

### M0.5 (Days 111-124) - APPROVED ✅
- Per-token step function refactor
- Metrics hooks (no-op)
- Basic Criterion benchmarks
- **Planning begins Day 110**

### M1 (Days 125+) - DEFERRED
- Paged KV cache (continuous batching)
- FlashAttention / CUDA Graphs
- Prefix cache (system prompt reuse)
- Prometheus metrics (wire M0.5 hooks)
- Performance test suite

---

## Clean Seams Requirement 🔧

**CRITICAL**: M0 implementation MUST use clean abstractions for M0.5/M1 integration

### 1. Inference Loop - Use Trait Abstraction

**Required**:
```rust
pub trait InferenceEngine {
    fn run(&mut self, prompt: &[u32], max_tokens: usize) -> Result<Vec<u32>>;
}

// M0: Simple loop implementation
pub struct SimpleInferenceEngine { /* ... */ }

// M0.5: Refactor to step-based (no API changes)
// M1: Add continuous batching
```

**Why**: M0.5 refactors internals to step-based, HTTP layer unchanged

---

### 2. KV Cache - Use Allocator Trait

**Required**:
```rust
pub trait KVCacheAllocator {
    fn allocate(&mut self, size: usize) -> Result<*mut u8>;
    fn free(&mut self, ptr: *mut u8) -> Result<()>;
}

// M0: Simple contiguous allocator
pub struct ContiguousAllocator { /* ... */ }

// M1: Paged allocator (drop-in replacement)
pub struct PagedAllocator { /* ... */ }
```

**Why**: M1 swaps allocator without changing inference loop

---

### 3. Attention - Use Kernel Trait

**Required**:
```rust
pub trait AttentionKernel {
    fn compute(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor>;
}

// M0: Basic attention
pub struct BasicAttention { /* ... */ }

// M1: FlashAttention (drop-in replacement)
pub struct FlashAttention { /* ... */ }
```

**Why**: M1 adds FlashAttention without refactoring inference

---

### 4. Metrics - Add Timing Measurements

**Required**:
```rust
pub async fn execute_inference(prompt: String) -> Result<Vec<u32>> {
    let start = Instant::now();
    
    let tokens = tokenize(prompt)?;
    let first_token_start = Instant::now();
    
    let first_token = generate_first_token(&tokens)?;
    let first_token_latency = first_token_start.elapsed();
    
    // M0: Just measure, don't emit
    // M0.5: Add MetricsHook::record_first_token_latency(first_token_latency)
    
    // Continue inference...
}
```

**Why**: M0.5 wires measurements to MetricsHook trait

---

### 5. CUDA Graph - Keep Decode Loop Pure

**Required**:
```rust
// BAD: Side effects in loop (can't CUDA Graph)
for _ in 0..max_tokens {
    log::info!("Generating token {}", i); // ❌ Side effect
    let token = forward_pass()?;
}

// GOOD: Pure loop (CUDA Graph ready)
for _ in 0..max_tokens {
    let token = forward_pass()?; // ✅ Pure
    tokens.push(token);
}
// Log after loop
log::info!("Generated {} tokens", tokens.len());
```

**Why**: M1 wraps decode loop in CUDA Graph

---

## Story-Specific Guidelines

### Foundation Team (Foundation-Alpha)

**FT-021 (KV Cache Allocation)**:
- ✅ Use `KVCacheAllocator` trait (not direct `cudaMalloc`)
- ✅ Implement `ContiguousAllocator` (M0)
- ✅ Document trait for `PagedAllocator` (M1)

**FT-016 (cuBLAS GEMM Wrapper)**:
- ✅ Isolate attention computation in `AttentionKernel` trait
- ✅ Implement `BasicAttention` (M0)
- ✅ Document trait for `FlashAttention` (M1)

**FT-002 (POST /execute Endpoint)**:
- ✅ Use `InferenceEngine` trait (not direct loop)
- ✅ Add timing measurements (even if unused)
- ✅ Keep HTTP layer decoupled from inference internals

**FT-018 (Sampling Kernel)**:
- ✅ Keep decode loop pure (no side effects)
- ✅ Document CUDA Graph readiness

---

### Llama Team (Llama-Beta)

**LT-016 (Llama Inference Loop)**:
- ✅ Implement `InferenceEngine` trait
- ✅ Keep loop modular (easy to refactor to step-based)

**LT-014 (GQA Attention)**:
- ✅ Implement `AttentionKernel` trait
- ✅ Isolate GQA computation

---

### GPT Team (GPT-Gamma)

**GT-020 (GPT Inference Loop)**:
- ✅ Implement `InferenceEngine` trait
- ✅ Keep loop modular (easy to refactor to step-based)

**GT-015 (MHA Attention)**:
- ✅ Implement `AttentionKernel` trait
- ✅ Isolate MHA computation

---

## Timeline

```
Day 1                Day 110              Day 124              Day 125+
  |                     |                    |                    |
  ↓                     ↓                    ↓                    ↓
M0 Start            M0 Complete         M0.5 Complete        M1 Start
(139 stories)       (3 models working)  (Perf foundation)    (Advanced perf)
  |                     |                    |                    |
  |------ 110 days -----|---- +14 days ------|                    |
  |                     |                    |                    |
Foundation-Alpha        |                    |                    |
Llama-Beta              |                    |                    |
GPT-Gamma               |                    |                    |
                        |                    |                    |
                    M0.5 Planning       M1 Planning              |
                    (1 day)             (TBD)                     |
```

**M0**: Days 1-110 (unchanged)  
**M0.5**: Days 111-124 (+2 weeks)  
**M1**: Days 125+ (continuous batching, FlashAttention, etc.)

---

## M0 Day 1 Launch - Action Items

### Foundation-Alpha (Monitor 1)
```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd/.plan/foundation-team

# Update day tracker
# Current Day: 1
# Current Story: FT-001 (HTTP Server Setup)
# Sprint: Sprint 1 - HTTP Foundation

# Open story card
code sprints/sprint-1-http-foundation/FT-001.md

# Launch: "You are Foundation-Alpha. Begin FT-001: HTTP Server Setup.
#         Review the story card and acceptance criteria.
#         IMPORTANT: Use trait abstractions (InferenceEngine, KVCacheAllocator, AttentionKernel).
#         Add timing measurements. Keep decode loop pure.
#         Work on this story until complete, then report completion."
```

### Llama-Beta (Monitor 2)
```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd/.plan/llama-team

# Update day tracker
# Current Day: 1
# Current Story: LT-000 (GGUF Format Research)
# Sprint: Sprint 0 - Prep Work
# Status: Waiting for FFI lock (day 11)

# Open story card
code sprints/sprint-0-prep-work/LT-000.md

# Launch: "You are Llama-Beta. Begin LT-000: GGUF Format Research.
#         Study llama.cpp implementation, GGUF format spec, design test framework.
#         You are blocked from GGUF loader work until day 11 (FFI lock).
#         Focus on research and preparation."
```

### GPT-Gamma (Monitor 3)
```bash
cd /home/vince/Projects/llama-orch/bin/worker-orcd/.plan/gpt-team

# Update day tracker
# Current Day: 1
# Current Story: GT-000 (MXFP4 Spec Study)
# Sprint: Sprint 0 - Prep Work
# Status: Waiting for FFI lock (day 11)

# Open story card
code sprints/sprint-0-prep-work/GT-000.md

# Launch: "You are GPT-Gamma. Begin GT-000: MXFP4 Spec Study.
#         Study MXFP4 format spec, HF tokenizers crate, design validation framework.
#         You are blocked from HF tokenizer work until day 11 (FFI lock).
#         Focus on research and preparation."
```

---

## M0.5 Planning (Day 110)

**When M0 reaches stability** (Gate 4 passed, all models working):

### Planning Artifacts to Create (1 day PM effort)

1. **3 Story Cards**:
   - M0.5-001: Per-Token Step Function Refactor (3 days)
   - M0.5-002: Metrics Hooks (No-Op) (2 days)
   - M0.5-003: Basic Criterion Benchmarks (2 days)

2. **1 Sprint README**:
   - `sprints/sprint-m0.5-performance-foundation/README.md`

3. **1 Gate Checklist**:
   - `integration-gates/gate-m0.5-performance-foundation.md`

4. **Execution Tracking**:
   - Update day-tracker.md (Days 111-124)
   - Update milestones.md (M0.5 milestone)

---

## Success Criteria

### M0 Success (Day 110)
- ✅ All 139 stories complete
- ✅ All 3 models working (Qwen, Phi-3, GPT-OSS-20B)
- ✅ All tests passing
- ✅ **Clean seams implemented** (traits, measurements, pure loops)
- ✅ Documentation complete
- ✅ Ready for M0.5 planning

### M0.5 Success (Day 124)
- ✅ Step function refactor complete (no API changes)
- ✅ Metrics hooks defined (no-op implementation)
- ✅ Criterion benchmarks running (baseline data)
- ✅ All M0 tests still passing
- ✅ Ready for M1 performance work

### M1 Success (Future)
- ✅ Paged KV cache (continuous batching)
- ✅ FlashAttention / CUDA Graphs (latency optimization)
- ✅ Prefix cache (system prompt reuse)
- ✅ Prometheus metrics (wire M0.5 hooks)
- ✅ Performance test suite (comprehensive validation)
- ✅ Competitive parity with llama.cpp/vLLM

---

## Questions?

**For clean seams guidance**:
- See: `.docs/DECISION_APPROVED_M0_WITH_M0.5_PLAN.md` (detailed guidelines)

**For M0 execution**:
- See: `.docs/EXECUTION_GUIDE.md` (day-by-day execution plan)

**For story details**:
- See: `{team}/stories/{STORY-ID}.md` (individual story cards)

**For dependencies**:
- See: `{team}/execution/dependencies.md` (upstream/downstream tracking)

---

## Key Reminders

1. **Clean seams are MANDATORY** - Use trait abstractions, not direct implementations
2. **Add timing measurements** - Even if unused in M0, M0.5 will wire them
3. **Keep decode loop pure** - No side effects, CUDA Graph ready
4. **Document extension points** - Note where M1 features integrate
5. **M0 scope unchanged** - 110 days, 139 stories, current plan

---

## Communication Channels

**Daily Standups**: Update `execution/day-tracker.md`  
**Blockers**: Update `execution/dependencies.md`, notify PM  
**Gate Validation**: Update `integration-gates/gate-N-*.md`  
**Coordination**: Use `coordination/` folder for cross-team docs

---

**Status**: M0 Day 1 Launch - PROCEED ✅  
**Next Milestone**: FFI Lock (Day 11)  
**Final Milestone**: M0 Complete (Day 110)  
**Prepared By**: Project Manager (M0 Worker-orcd) 📋

---

Planned by Project Management Team 📋
