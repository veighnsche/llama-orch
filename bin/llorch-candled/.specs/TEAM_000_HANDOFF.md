# TEAM-000 Foundation Handoff

**Created by:** TEAM-000 (Foundation)  
**Date:** 2025-10-08  
**Purpose:** Handoff of llorch-candled scaffolding to implementation teams  
**Status:** ✅ Scaffolding Complete

---

## What We Built

TEAM-000 has created the **complete scaffolding** for `llorch-candled`, a Candle-based Llama-2 inference worker following the hybrid approach recommended in `CANDLE_INTEGRATION_HANDOFF.md`.

### Deliverables

1. ✅ **Project Structure** - Complete directory layout
2. ✅ **Cargo.toml** - All dependencies configured (worker-crates + ndarray + candle-kernels)
3. ✅ **HTTP Server** - main.rs with worker-http integration
4. ✅ **Backend Stub** - CandleInferenceBackend implementing InferenceBackend trait
5. ✅ **Module Scaffolding** - All layer modules with stubs
6. ✅ **Checkpoint 0** - Foundation validation spec
7. ✅ **Tests** - Foundation test suite
8. ✅ **Documentation** - README and handoff docs
9. ✅ **Workspace Integration** - Added to root Cargo.toml

---

## Architecture Overview

### Hybrid Approach (from CANDLE_INTEGRATION_HANDOFF.md)

**✅ RECOMMENDED:** Use Candle's low-level components, not the full framework  
**❌ NOT RECOMMENDED:** Using full Candle framework (too much abstraction)  
**✅ KEEP:** Checkpoint-driven validation approach  
**✅ KEEP:** HTTP server architecture

### Implementation Strategy

```
Week 2 (Core Layers):
├─ Checkpoint 1: RMSNorm (CPU with ndarray)
├─ Checkpoint 1B: RoPE (NEW for Llama-2)
├─ Checkpoint 2: Separate QKV (not combined like GPT-2)
├─ Checkpoint 3: KV Cache (32 layers, 4096 context)
└─ Checkpoint 6: SwiGLU (not GELU)

Week 3 (Full Model):
├─ Checkpoint 4: Attention Scores (scale = 1/sqrt(128))
├─ Checkpoint 5: Attention Output (no bias)
├─ Checkpoint 7: First Block (RMSNorm + RoPE + SwiGLU)
└─ Checkpoint 8: Full Logits (32 layers, 32K vocab)

Week 4 (Validation & CUDA):
├─ Checkpoints 9-11: Sampling
├─ Checkpoint 12: End-to-End
└─ CUDA kernel integration (optional)
```

---

## File Structure

```
bin/llorch-candled/
├── Cargo.toml                          # ✅ Complete with all deps
├── README.md                           # ✅ Full documentation
├── src/
│   ├── main.rs                         # ✅ HTTP server entry point
│   ├── lib.rs                          # ✅ Library exports
│   ├── backend/
│   │   ├── mod.rs                      # ✅ Module exports
│   │   └── candle_backend.rs           # ✅ Stub implementation
│   ├── model/
│   │   ├── mod.rs                      # ✅ Module exports
│   │   └── llama2.rs                   # ⏳ Stub (implement in Week 3)
│   ├── layers/
│   │   ├── mod.rs                      # ✅ Module exports
│   │   ├── rms_norm.rs                 # ⏳ Stub (Checkpoint 1)
│   │   ├── rope.rs                     # ⏳ Stub (Checkpoint 1B)
│   │   ├── embedding.rs                # ⏳ Stub (after basics)
│   │   ├── attention/
│   │   │   ├── mod.rs                  # ✅ Module exports
│   │   │   ├── qkv.rs                  # ⏳ Stub (Checkpoint 2)
│   │   │   ├── scores.rs               # ⏳ Stub (Checkpoint 4)
│   │   │   └── output.rs               # ⏳ Stub (Checkpoint 5)
│   │   ├── swiglu.rs                   # ⏳ Stub (Checkpoint 6)
│   │   └── transformer.rs              # ⏳ Stub (Checkpoint 7)
│   ├── cache/
│   │   ├── mod.rs                      # ✅ Module exports
│   │   └── kv_cache.rs                 # ⏳ Stub (Checkpoint 3)
│   ├── tensor/
│   │   ├── mod.rs                      # ✅ Module exports
│   │   └── ops.rs                      # ⏳ Stub (helpers)
│   └── error.rs                        # ✅ Error types
├── tests/
│   └── checkpoint_00_foundation.rs     # ✅ Foundation tests
└── .specs/
    ├── TEAM_000_HANDOFF.md             # ✅ This document
    └── checkpoints/
        └── CHECKPOINT_00_FOUNDATION.md # ✅ Foundation spec
```

---

## Key Design Decisions

### 1. Hybrid Compute Strategy

**CPU Path (Primary for MVP):**
- Pure ndarray implementation
- Used for checkpoint validation
- Educational value maintained
- Always works (no GPU required)

**CUDA Path (Optional for Performance):**
- Candle kernels only (not framework)
- Feature-gated with `cuda` feature
- Used after CPU validation passes
- Optimized performance

### 2. Module Organization

**Top-level cache module:**
- Signals future optimization work
- Room for paged attention, memory pooling
- Keep implementation simple for MVP

**Attention subdirectory:**
- Split into focused files (qkv, scores, output)
- Each file maps to a checkpoint
- Clear separation of concerns

### 3. Worker Crates Integration

**100% Reusable:**
- `worker-common`: SamplingConfig, InferenceResult
- `worker-http`: HTTP server, SSE streaming
- `worker-tokenizer`: BPE tokenization
- `worker-models`: Model configs
- `worker-gguf`: GGUF parsing

**No reinvention:** Leverage existing infrastructure

### 4. Candle Integration

**What we use:**
- `candle-kernels`: CUDA kernels (optional)
- `cudarc`: CUDA runtime (optional)

**What we DON'T use:**
- `candle-core`: Too much abstraction
- `candle-nn`: We build our own layers
- `candle-transformers`: Defeats learning purpose

**Why:** Best performance without losing control

---

## Checkpoint-Driven Development

### Checkpoint 0: Foundation ✅ COMPLETE

**What's done:**
- Project structure created
- HTTP server wired up
- Worker crates integrated
- Stub backend implemented
- Tests written

**What's validated:**
- Compilation succeeds
- HTTP server starts
- Endpoints respond
- Tests pass

**Next:** Proceed to Checkpoint 1 (RMSNorm)

### Remaining Checkpoints

| # | Component | File | Status |
|---|-----------|------|--------|
| 1 | RMSNorm | `layers/rms_norm.rs` | ⏳ Ready for impl |
| 1B | RoPE | `layers/rope.rs` | ⏳ Ready for impl |
| 2 | QKV | `layers/attention/qkv.rs` | ⏳ Ready for impl |
| 3 | KV Cache | `cache/kv_cache.rs` | ⏳ Ready for impl |
| 4 | Scores | `layers/attention/scores.rs` | ⏳ Ready for impl |
| 5 | Output | `layers/attention/output.rs` | ⏳ Ready for impl |
| 6 | SwiGLU | `layers/swiglu.rs` | ⏳ Ready for impl |
| 7 | Block | `layers/transformer.rs` | ⏳ Ready for impl |
| 8 | Logits | `model/llama2.rs` | ⏳ Ready for impl |
| 9-11 | Sampling | `model/llama2.rs` | ⏳ Ready for impl |
| 12 | End-to-End | Full pipeline | ⏳ Ready for impl |

---

## How to Continue

### Step 1: Validate Foundation (Checkpoint 0)

```bash
cd bin/llorch-candled

# Build
cargo build

# Run tests
cargo test

# Start server (test mode)
cargo run -- \
  --worker-id test \
  --model test.gguf \
  --port 8080 \
  --callback-url http://localhost:9999

# Test health endpoint
curl http://localhost:8080/health
```

**Expected:** All tests pass, server starts, endpoints respond

### Step 2: Implement RMSNorm (Checkpoint 1)

1. Read `.specs/checkpoints/CHECKPOINT_01_RMS_NORM.md` (from llorch-cpud)
2. Implement `layers/rms_norm.rs` CPU path
3. Write test in `tests/checkpoint_01_rms_norm.rs`
4. Validate against llama.cpp checkpoint
5. Mark Checkpoint 1 as complete

### Step 3: Continue Through Checkpoints

- Follow checkpoint order (1 → 1B → 2 → 3 → 6 → 4 → 5 → 7 → 8 → 9-11 → 12)
- Validate each checkpoint before moving to next
- Use llama.cpp as reference implementation
- Keep CPU path working (CUDA is optional)

### Step 4: Add CUDA Acceleration (Week 3-4)

After CPU validation passes:

1. Enable `cuda` feature in Cargo.toml
2. Add CUDA kernel path to layers
3. Validate CUDA produces same output as CPU
4. Benchmark performance improvement

---

## Dependencies

### Already Configured ✅

```toml
[dependencies]
# Worker crates (100% reusable)
worker-common = { path = "../worker-crates/worker-common" }
worker-http = { path = "../worker-crates/worker-http" }
worker-tokenizer = { path = "../worker-crates/worker-tokenizer" }
worker-models = { path = "../worker-crates/worker-models" }
worker-gguf = { path = "../worker-crates/worker-gguf" }

# Candle integration (optional)
candle-kernels = { path = "../../reference/candle/candle-kernels", optional = true }
cudarc = { version = "0.11", optional = true }

# CPU compute
ndarray = "0.15"

# Async runtime (single-threaded!)
tokio = { version = "1", features = ["rt", "macros", "sync", "time"] }
async-trait = "0.1"

# Utilities
anyhow = "1.0"
thiserror = "1.0"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["json", "env-filter"] }
clap = { version = "4", features = ["derive"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
rand = "0.8"

[features]
cuda = ["candle-kernels", "cudarc"]
```

### To Add Later

- `approx`: For floating-point comparisons in tests
- `ndarray-npy`: For loading reference weights (if needed)

---

## Testing Strategy

### Unit Tests
- Test each layer in isolation
- Compare with reference implementation
- Validate shapes and values

### Checkpoint Tests
- One test file per checkpoint
- Load reference data from llama.cpp
- Assert outputs match within tolerance

### Integration Tests
- End-to-end inference
- HTTP server integration
- Worker crates integration

### Benchmarks
- CPU vs CUDA performance
- Tokens per second
- Memory usage

---

## Reference Implementation

**Primary:** llama.cpp
- Use for checkpoint validation
- Extract intermediate values
- Compare outputs

**Secondary:** Mistral.rs
- Reference for architecture patterns
- Study quantization approach
- Learn optimization techniques

**Tools:**
- TEAM-006's checkpoint extractor
- llama.cpp instrumentation
- Custom validation scripts

---

## Common Pitfalls to Avoid

### ❌ DON'T:
1. Use full Candle framework (defeats purpose)
2. Skip checkpoint validation
3. Implement CUDA before CPU works
4. Change worker-crates (they're reusable)
5. Create multiple .md files for same topic

### ✅ DO:
1. Follow checkpoint order strictly
2. Validate with llama.cpp reference
3. Keep CPU path working
4. Add TEAM-000 signatures to code
5. Update existing docs, don't create new ones

---

## Next Team Responsibilities

### Immediate (Week 2):
1. Validate Checkpoint 0 passes
2. Implement Checkpoint 1 (RMSNorm)
3. Implement Checkpoint 1B (RoPE)
4. Implement Checkpoint 2 (QKV)
5. Implement Checkpoint 6 (SwiGLU)

### Short-term (Week 3):
1. Implement Checkpoints 3-5 (Cache, Attention)
2. Implement Checkpoint 7 (Block)
3. Implement Checkpoint 8 (Full model)

### Long-term (Week 4):
1. Implement Checkpoints 9-12 (Sampling, E2E)
2. Add CUDA acceleration
3. Performance optimization
4. Production readiness

---

## Success Criteria

### Foundation (Checkpoint 0) ✅
- [x] Project compiles
- [x] HTTP server starts
- [x] Tests pass
- [x] Stub endpoints work
- [x] Worker crates integrated
- [x] Documentation complete

### MVP (Checkpoints 1-12) ⏳
- [ ] All checkpoints pass
- [ ] CPU inference works
- [ ] Outputs match llama.cpp
- [ ] End-to-end test passes

### Production (Future) ⏳
- [ ] CUDA acceleration working
- [ ] Performance benchmarked
- [ ] Memory optimized
- [ ] Production deployed

---

## Files to Reference

### In llorch-candled:
- `README.md`: Overview and quick start
- `.specs/checkpoints/CHECKPOINT_00_FOUNDATION.md`: Foundation spec
- `Cargo.toml`: Dependencies and features
- `src/`: All source code (with stubs and TODOs)

### In llorch-cpud:
- `.specs/checkpoints/`: All checkpoint specs (adapt for Llama-2)
- `.specs/CANDLE_INTEGRATION_HANDOFF.md`: Integration strategy
- `.specs/checkpoints/LLAMA2_CHECKPOINT_UPDATE_PLAN.md`: Checkpoint updates

### In reference/:
- `candle/candle-kernels/`: CUDA kernels source
- `mistral.rs/`: Reference implementation

---

## Questions & Support

### For Architecture Questions:
- Read `CANDLE_INTEGRATION_HANDOFF.md`
- Study Mistral.rs implementation
- Check llorch-cpud for patterns

### For Checkpoint Questions:
- Read checkpoint spec in `.specs/checkpoints/`
- Compare with llama.cpp reference
- Use TEAM-006's checkpoint extractor

### For Integration Questions:
- Check worker-crates documentation
- Review llorch-cpud integration
- Test with stub endpoints first

---

## Sign-off

**Created by:** TEAM-000 (Foundation)  
**Date:** 2025-10-08  
**Status:** ✅ Scaffolding Complete, Ready for Implementation

**Deliverables:**
- ✅ Complete project structure
- ✅ All dependencies configured
- ✅ HTTP server integrated
- ✅ Stub implementation working
- ✅ Tests passing
- ✅ Documentation complete
- ✅ Workspace integrated

**Next Steps:**
1. Validate Checkpoint 0
2. Implement Checkpoint 1 (RMSNorm)
3. Continue through checkpoints
4. Add CUDA acceleration (Week 3-4)

---

*"The foundation of them all."*  
— TEAM-000, Foundation Implementation Division

**END HANDOFF**
