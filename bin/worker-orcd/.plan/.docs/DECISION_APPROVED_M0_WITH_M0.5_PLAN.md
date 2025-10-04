# Decision Approved: M0 + M0.5 Performance Foundation

**Date**: 2025-10-04 02:42  
**Decision**: ✅ **APPROVED** - Proceed with M0 as planned, M0.5 to follow  
**Approved By**: Management / Stakeholders

---

## Executive Summary

**Decision**: 
- ✅ Proceed with **M0 as planned** (110 days, 139 stories, current scope)
- ✅ **M0.5 approved** for post-M0 (+2 weeks, performance foundation)
- ✅ Structure M0 with **clean seams** for M0.5/M1 performance work

**Timeline**:
- **M0**: Days 1-110 (current plan, unchanged)
- **M0.5**: Days 111-124 (+14 days, performance foundation)
- **M1**: Days 125+ (continuous batching, advanced features)

---

## M0 Scope (Days 1-110) - UNCHANGED ✅

### Core Functionality
- ✅ 3 models: Qwen2.5-0.5B, Phi-3-Mini, GPT-OSS-20B
- ✅ 3 quantization formats: Q4_K_M, MXFP4, Q4_0
- ✅ 2 tokenizer backends: GGUF byte-BPE, tokenizer.json
- ✅ Architecture adapters: LlamaInferenceAdapter + GPTInferenceAdapter
- ✅ Architecture detection from GGUF metadata

### Critical Safety
- ✅ VRAM OOM handling (M0-W-1021)
- ✅ VRAM residency verification (M0-W-1012)
- ✅ Kernel safety validation (M0-W-1431)

### Correctness
- ✅ MXFP4 numerical correctness validation (M0-W-1822)
- ✅ Minimal same-device reproducibility check (seeded RNG, temp=0)

### Planning Status
- ✅ 139 story cards ready
- ✅ 24 sprint READMEs ready
- ✅ 12 gate checklists ready
- ✅ 189 total artifacts ready
- ✅ **Day 1 launch: PROCEED**

---

## M0.5 Scope (Days 111-124) - APPROVED ✅

### Performance Foundation (3 Features)

#### 1. Per-Token Step Function Refactor
**Purpose**: Refactor inference loop to step-based architecture

**Implementation**:
```rust
pub trait InferenceStep {
    fn start(&mut self, prompt: &[u32]) -> Result<()>;
    fn next_token(&mut self) -> Result<u32>;
    fn free(&mut self) -> Result<()>;
}
```

**Benefits**:
- ✅ Clean seam for continuous batching (M1)
- ✅ No API changes to HTTP layer
- ✅ Localized to inference loop

**Story**: M0.5-001 (3 days)

---

#### 2. Metrics Hooks (No-Op)
**Purpose**: Add performance metric collection points (no-op in M0.5, wired in M1)

**Implementation**:
```rust
pub trait MetricsHook {
    fn record_first_token_latency(&self, ms: f64) {}
    fn record_token_generation(&self, token_id: u32) {}
    fn record_inference_complete(&self, total_ms: f64) {}
}

// M0.5: No-op implementation
pub struct NoOpMetrics;
impl MetricsHook for NoOpMetrics {}

// M1: Prometheus implementation
pub struct PrometheusMetrics { /* ... */ }
impl MetricsHook for PrometheusMetrics { /* ... */ }
```

**Benefits**:
- ✅ API surface defined in M0.5
- ✅ Zero overhead (no-op)
- ✅ M1 wires Prometheus without refactor

**Story**: M0.5-002 (2 days)

---

#### 3. Basic Criterion Benchmarks
**Purpose**: Establish performance baseline with minimal benchmark suite

**Implementation**:
```rust
// benches/inference_baseline.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_qwen_haiku(c: &mut Criterion) {
    c.bench_function("qwen_haiku_inference", |b| {
        b.iter(|| {
            // Load model, run haiku prompt, measure
            black_box(run_inference("qwen2.5-0.5b", HAIKU_PROMPT))
        });
    });
}

criterion_group!(benches, benchmark_qwen_haiku);
criterion_main!(benches);
```

**Metrics Collected**:
- First-token latency (ms)
- Per-token latency (ms)
- Total inference time (ms)
- Tokens/second

**Benefits**:
- ✅ Baseline data for M1 optimization
- ✅ Regression detection
- ✅ Minimal maintenance overhead

**Story**: M0.5-003 (2 days)

---

### M0.5 Timeline

**Duration**: 14 days (2 weeks)

**Sprint Plan**:
- **Days 111-113**: M0.5-001 (Step function refactor)
- **Days 114-115**: M0.5-002 (Metrics hooks)
- **Days 116-117**: M0.5-003 (Criterion benchmarks)
- **Days 118-120**: Integration testing
- **Days 121-123**: Documentation
- **Day 124**: M0.5 Gate validation

**Gate Criteria**:
- [ ] Step function refactor complete (no API changes)
- [ ] Metrics hooks defined (no-op implementation)
- [ ] Criterion benchmarks running (baseline data collected)
- [ ] All M0 tests still passing
- [ ] Documentation updated

---

## M1 Scope (Days 125+) - DEFERRED

### Advanced Performance Features

**From Performance Team Proposal** (deferred from M0):
1. ✅ Paged KV Cache Block Allocator (continuous batching prerequisite)
2. ✅ FlashAttention / CUDA Graphs (latency optimization)
3. ✅ Prefix Cache (system prompt KV reuse)
4. ✅ Prometheus metrics endpoint (wire M0.5 hooks)
5. ✅ Performance test suite (comprehensive validation)

**Additional M1 Features**:
- Continuous batching / rolling admission
- Chunked prefill (long prompts)
- Quantization variants (INT8, AWQ)
- Speculative decoding (optional)
- Scheduling policy knobs

**M1 Planning**: Post-M0.5 completion

---

## Clean Seams Strategy

### Architectural Principles for M0

To ensure M0.5/M1 performance work integrates cleanly, M0 implementation MUST follow:

#### 1. Inference Loop Abstraction
**M0 Requirement**: Keep inference loop modular

**Bad** (tightly coupled):
```rust
pub async fn execute_inference(prompt: String) -> Result<Vec<u32>> {
    let tokens = tokenize(prompt)?;
    let mut output = Vec::new();
    
    // Tightly coupled: hard to refactor to step-based
    for _ in 0..max_tokens {
        let logits = forward_pass(&tokens)?;
        let token = sample(logits)?;
        output.push(token);
        tokens.push(token);
    }
    
    Ok(output)
}
```

**Good** (clean seam):
```rust
pub trait InferenceEngine {
    fn run(&mut self, prompt: &[u32], max_tokens: usize) -> Result<Vec<u32>>;
}

// M0: Simple loop implementation
pub struct SimpleInferenceEngine { /* ... */ }

impl InferenceEngine for SimpleInferenceEngine {
    fn run(&mut self, prompt: &[u32], max_tokens: usize) -> Result<Vec<u32>> {
        // M0.5 can refactor this to step-based without changing trait
        // ...
    }
}
```

**Benefit**: M0.5 refactors `SimpleInferenceEngine` internals, HTTP layer unchanged

---

#### 2. Metrics Collection Points
**M0 Requirement**: Add timing measurements (even if unused)

**Implementation**:
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

**Benefit**: M0.5 adds `MetricsHook` trait, wires to existing measurements

---

#### 3. KV Cache Interface
**M0 Requirement**: Abstract KV cache allocation

**Bad** (direct allocation):
```rust
let kv_cache = cudaMalloc(kv_cache_size)?; // Hard to replace with paged allocator
```

**Good** (abstracted):
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

**Benefit**: M1 swaps `ContiguousAllocator` for `PagedAllocator`, no inference loop changes

---

#### 4. Attention Kernel Path
**M0 Requirement**: Isolate attention computation

**Implementation**:
```rust
pub trait AttentionKernel {
    fn compute(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor>;
}

// M0: Basic attention
pub struct BasicAttention { /* ... */ }

// M1: FlashAttention (drop-in replacement)
pub struct FlashAttention { /* ... */ }
```

**Benefit**: M1 adds `FlashAttention` implementation, selects via config

---

#### 5. CUDA Graph Readiness
**M0 Requirement**: Keep decode loop pure (no side effects)

**Bad** (side effects in loop):
```rust
for _ in 0..max_tokens {
    log::info!("Generating token {}", i); // Side effect: can't CUDA Graph
    let token = forward_pass()?;
}
```

**Good** (pure loop):
```rust
for _ in 0..max_tokens {
    let token = forward_pass()?; // Pure: can wrap in CUDA Graph
    tokens.push(token);
}
// Log after loop
log::info!("Generated {} tokens", tokens.len());
```

**Benefit**: M1 wraps decode loop in CUDA Graph without refactor

---

## M0 Implementation Guidelines

### For Foundation Team (Foundation-Alpha)

**Stories with Clean Seam Requirements**:

1. **FT-021 (KV Cache Allocation)**:
   - ✅ Use `KVCacheAllocator` trait (not direct `cudaMalloc`)
   - ✅ Implement `ContiguousAllocator` (M0)
   - ✅ Document trait for `PagedAllocator` (M1)

2. **FT-016 (cuBLAS GEMM Wrapper)**:
   - ✅ Isolate attention computation in trait
   - ✅ Implement `BasicAttention` (M0)
   - ✅ Document trait for `FlashAttention` (M1)

3. **FT-002 (POST /execute Endpoint)**:
   - ✅ Use `InferenceEngine` trait (not direct loop)
   - ✅ Add timing measurements (even if unused)
   - ✅ Keep HTTP layer decoupled from inference internals

4. **FT-018 (Sampling Kernel)**:
   - ✅ Keep decode loop pure (no side effects)
   - ✅ Document CUDA Graph readiness

---

### For Llama Team (Llama-Beta)

**Stories with Clean Seam Requirements**:

1. **LT-016 (Llama Inference Loop)**:
   - ✅ Implement `InferenceEngine` trait
   - ✅ Keep loop modular (easy to refactor to step-based)

2. **LT-014 (GQA Attention)**:
   - ✅ Implement `AttentionKernel` trait
   - ✅ Isolate GQA computation

---

### For GPT Team (GPT-Gamma)

**Stories with Clean Seam Requirements**:

1. **GT-020 (GPT Inference Loop)**:
   - ✅ Implement `InferenceEngine` trait
   - ✅ Keep loop modular (easy to refactor to step-based)

2. **GT-015 (MHA Attention)**:
   - ✅ Implement `AttentionKernel` trait
   - ✅ Isolate MHA computation

---

## M0.5 Planning (Post-M0)

### When to Plan M0.5

**Trigger**: M0 reaches stability
- ✅ Gate 4 (Day 110) passed
- ✅ All 3 models working (Qwen, Phi-3, GPT-OSS-20B)
- ✅ All tests passing
- ✅ Documentation complete

**Planning Timeline**:
- **Day 110**: M0 complete
- **Days 110-111**: M0.5 planning (1 day PM effort)
- **Day 111**: M0.5 Day 1 launch

---

### M0.5 Planning Artifacts

**To Create** (1 day PM effort):

1. **3 Story Cards**:
   - M0.5-001: Per-Token Step Function Refactor (3 days)
   - M0.5-002: Metrics Hooks (No-Op) (2 days)
   - M0.5-003: Basic Criterion Benchmarks (2 days)

2. **1 Sprint README**:
   - `sprints/sprint-m0.5-performance-foundation/README.md`
   - Days 111-124 execution plan

3. **1 Gate Checklist**:
   - `integration-gates/gate-m0.5-performance-foundation.md`
   - Validation criteria

4. **Execution Tracking**:
   - Update `execution/day-tracker.md` (Days 111-124)
   - Update `execution/milestones.md` (M0.5 milestone)

---

## Communication Plan

### To Performance Team

**Message**:
```
Hi Performance Team,

Great news! Management has approved the M0.5 performance foundation plan.

**Decision**:
- ✅ M0 proceeds as planned (Days 1-110, current scope)
- ✅ M0.5 approved (Days 111-124, +2 weeks)
- ✅ M0.5 scope: Step function refactor, metrics hooks, Criterion benchmarks
- ✅ M1 scope: All 5 original features (paged KV cache, FlashAttention, prefix cache, etc.)

**M0 Implementation**:
We'll structure M0 with clean seams for M0.5/M1:
- InferenceEngine trait (easy step function refactor)
- KVCacheAllocator trait (easy paged allocator swap)
- AttentionKernel trait (easy FlashAttention integration)
- Timing measurements in place (easy metrics wiring)
- Pure decode loop (CUDA Graph ready)

**Next Steps**:
1. M0 Day 1 launch proceeds (current scope)
2. M0.5 planning begins Day 110 (1 day PM effort)
3. M0.5 execution Days 111-124 (+2 weeks)
4. M1 planning post-M0.5

Thank you for the detailed proposal. The clean seams approach ensures we get performance foundation without disrupting M0 delivery.

Best,
PM (M0 Worker-orcd) 📋
```

---

### To All Teams (Foundation, Llama, GPT)

**Message**:
```
Hi Teams,

**M0 Day 1 Launch: PROCEED**

M0 scope is confirmed and unchanged. However, we have M0.5 performance foundation approved for post-M0 (+2 weeks).

**Clean Seams Requirement**:
To ensure M0.5/M1 performance work integrates cleanly, please follow these guidelines:

1. **Use trait abstractions**:
   - InferenceEngine (not direct loop)
   - KVCacheAllocator (not direct cudaMalloc)
   - AttentionKernel (not direct computation)

2. **Add timing measurements**:
   - Measure first-token latency (even if unused)
   - Measure per-token latency (even if unused)
   - M0.5 will wire these to MetricsHook

3. **Keep decode loop pure**:
   - No side effects in loop (CUDA Graph readiness)
   - Log before/after loop, not during

4. **Document extension points**:
   - Note where PagedAllocator replaces ContiguousAllocator
   - Note where FlashAttention replaces BasicAttention
   - Note where step-based refactor happens

**See**: `.docs/DECISION_APPROVED_M0_WITH_M0.5_PLAN.md` for detailed guidelines

**M0 Timeline** (unchanged):
- Foundation-Alpha: Days 1-89
- Llama-Beta: Days 1-90
- GPT-Gamma: Days 1-110

**M0.5 Timeline** (new):
- All teams: Days 111-124 (+2 weeks)

Let's deliver M0 with clean seams for performance work!

Best,
PM (M0 Worker-orcd) 📋
```

---

## Success Criteria

### M0 Success (Day 110)
- ✅ All 139 stories complete
- ✅ All 3 models working (Qwen, Phi-3, GPT-OSS-20B)
- ✅ All tests passing
- ✅ Clean seams implemented (traits, measurements, pure loops)
- ✅ Documentation complete
- ✅ **Ready for M0.5 planning**

### M0.5 Success (Day 124)
- ✅ Step function refactor complete (no API changes)
- ✅ Metrics hooks defined (no-op implementation)
- ✅ Criterion benchmarks running (baseline data)
- ✅ All M0 tests still passing
- ✅ **Ready for M1 performance work**

### M1 Success (Future)
- ✅ Paged KV cache (continuous batching)
- ✅ FlashAttention / CUDA Graphs (latency optimization)
- ✅ Prefix cache (system prompt reuse)
- ✅ Prometheus metrics (wire M0.5 hooks)
- ✅ Performance test suite (comprehensive validation)
- ✅ **Competitive parity with llama.cpp/vLLM**

---

## Action Items

### Immediate (PM)
1. ✅ Document decision (this file)
2. ✅ Notify Performance team (approval message)
3. ✅ Notify all teams (clean seams guidelines)
4. ✅ Update execution tracking (M0.5 milestone added)
5. ✅ **Proceed with M0 Day 1 launch**

### Day 110 (Post-M0)
1. ⬜ Create M0.5 planning artifacts (3 stories, 1 sprint, 1 gate)
2. ⬜ Update execution tracking (Days 111-124)
3. ⬜ Launch M0.5 Day 1 (Day 111)

### Day 124 (Post-M0.5)
1. ⬜ Validate M0.5 gate (performance foundation complete)
2. ⬜ Begin M1 planning (continuous batching, FlashAttention, etc.)

---

## Timeline Summary

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
                                                                  |
                                                        Continuous batching
                                                        FlashAttention
                                                        Prefix cache
                                                        Prometheus metrics
                                                        Performance tests
```

**Total M0 + M0.5**: 124 days (17.7 weeks)  
**M0 alone**: 110 days (15.7 weeks) ← unchanged  
**M0.5 addition**: +14 days (+2 weeks)

---

## Conclusion

**Decision**: ✅ **APPROVED**

**M0 Scope**: Unchanged (110 days, 139 stories, current scope)

**M0.5 Scope**: Approved (+2 weeks, performance foundation)
- Per-token step function refactor
- Metrics hooks (no-op)
- Basic Criterion benchmarks

**M1 Scope**: Deferred (continuous batching, FlashAttention, prefix cache, etc.)

**Clean Seams**: Required in M0 implementation
- Trait abstractions (InferenceEngine, KVCacheAllocator, AttentionKernel)
- Timing measurements (even if unused)
- Pure decode loop (CUDA Graph ready)

**Next Action**: **Proceed with M0 Day 1 launch** ✅

---

**Status**: Decision Documented, Teams Notified  
**Next Milestone**: M0 Day 1 Launch (Immediate)  
**Prepared By**: Project Manager (M0 Worker-orcd) 📋  
**Date**: 2025-10-04 02:42

---

Planned by Project Management Team 📋
