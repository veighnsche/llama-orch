# Worker Crates Scaffold — Complete

**Date**: 2025-10-05  
**Status**: ✅ Complete  
**Next Steps**: Phase 1 extraction from worker-orcd

---

## Summary

Successfully scaffolded 6 shared worker crates in `bin/worker-crates/` to enable 85% code reuse between platform-specific workers (NVIDIA CUDA, Apple ARM Metal, etc.).

## Crates Created

### 1. worker-http
**Path**: `bin/worker-crates/worker-http/`  
**Purpose**: HTTP server, SSE streaming, route handling  
**Status**: ✅ Scaffold complete, compiles  
**Extraction Source**: `bin/worker-orcd/src/http/`

### 2. worker-gguf
**Path**: `bin/worker-crates/worker-gguf/`  
**Purpose**: GGUF file format parser (pure Rust)  
**Status**: ✅ Scaffold complete, compiles  
**Extraction Source**: `bin/worker-orcd/src/gguf/mod.rs` (277 lines)

### 3. worker-tokenizer
**Path**: `bin/worker-crates/worker-tokenizer/`  
**Purpose**: Tokenization (BPE, HuggingFace)  
**Status**: ✅ Scaffold complete, compiles  
**Extraction Source**: `bin/worker-orcd/src/tokenizer/` (~1200 lines)

### 4. worker-models
**Path**: `bin/worker-crates/worker-models/`  
**Purpose**: Model adapters (GPT, Llama, Phi-3, Qwen)  
**Status**: ✅ Scaffold complete, compiles  
**Extraction Source**: `bin/worker-orcd/src/models/` (~800 lines)

### 5. worker-common
**Path**: `bin/worker-crates/worker-common/`  
**Purpose**: Common types, errors, sampling config  
**Status**: ✅ Scaffold complete, compiles  
**Extraction Source**: `bin/worker-orcd/src/{error.rs,sampling_config.rs,inference_result.rs,startup.rs}`

### 6. worker-compute
**Path**: `bin/worker-crates/worker-compute/`  
**Purpose**: Platform-agnostic compute trait  
**Status**: ✅ Scaffold complete, trait defined, compiles  
**Extraction Source**: New abstraction layer

---

## Root Cargo.toml Updated

Added new section to workspace members:

```toml
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# WORKER CRATES — Shared Worker Components
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared libraries used across all worker implementations (worker-orcd, worker-aarmd).
# These crates contain pure Rust code with no platform-specific FFI dependencies.

"bin/worker-crates/worker-http",       # HTTP server, SSE streaming, route handling
"bin/worker-crates/worker-gguf",       # GGUF file format parser (pure Rust)
"bin/worker-crates/worker-tokenizer",  # Tokenization (BPE, HuggingFace)
"bin/worker-crates/worker-models",     # Model adapters (GPT, Llama, Phi-3, Qwen)
"bin/worker-crates/worker-common",     # Common types, errors, sampling config
"bin/worker-crates/worker-compute",    # Platform-agnostic compute trait
```

---

## Verification

```bash
$ cargo check -p worker-http -p worker-gguf -p worker-tokenizer \
              -p worker-models -p worker-common -p worker-compute

✅ All crates compile successfully
```

---

## Directory Structure

```
bin/worker-crates/
├── README.md                      # Overview and usage guide
├── worker-http/
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs                 # Placeholder (extraction pending)
├── worker-gguf/
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs                 # Placeholder (extraction pending)
├── worker-tokenizer/
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs                 # Placeholder (extraction pending)
├── worker-models/
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs                 # Placeholder (extraction pending)
├── worker-common/
│   ├── Cargo.toml
│   └── src/
│       └── lib.rs                 # Placeholder (extraction pending)
└── worker-compute/
    ├── Cargo.toml
    └── src/
        └── lib.rs                 # ComputeBackend trait defined
```

---

## Key Design Decisions

### 1. Pure Rust Shared Crates
All shared crates contain **zero FFI code**. Platform-specific FFI (CUDA, Metal) remains in worker binaries.

### 2. ComputeBackend Trait
Defined platform-agnostic trait for compute operations:

```rust
pub trait ComputeBackend {
    type Context;
    type Model;
    type InferenceResult;
    
    fn init(device_id: i32) -> Result<Self::Context, ComputeError>;
    fn load_model(ctx: &Self::Context, path: &str) -> Result<Self::Model, ComputeError>;
    fn inference_start(...) -> Result<Self::InferenceResult, ComputeError>;
    fn inference_next_token(...) -> Result<Option<String>, ComputeError>;
    fn get_memory_usage(model: &Self::Model) -> u64;
    fn memory_architecture() -> &'static str;
}
```

### 3. Placeholder Pattern
Each crate has a placeholder module until extraction is complete. This allows:
- ✅ Workspace compiles immediately
- ✅ Dependencies can be added incrementally
- ✅ No broken build states

---

## Next Steps (Phase 1: Extraction)

### 1. Extract worker-gguf (2 hours)
```bash
# Copy from worker-orcd
cp bin/worker-orcd/src/gguf/mod.rs bin/worker-crates/worker-gguf/src/lib.rs

# Update imports
# Test
cargo test -p worker-gguf
```

**Estimated**: 2 hours  
**Complexity**: Low (already pure Rust, 277 lines)

### 2. Extract worker-tokenizer (6 hours)
```bash
# Copy all tokenizer modules
cp -r bin/worker-orcd/src/tokenizer/* bin/worker-crates/worker-tokenizer/src/

# Update lib.rs
# Update imports
# Test
cargo test -p worker-tokenizer
```

**Estimated**: 6 hours  
**Complexity**: Medium (~1200 lines, multiple modules)

### 3. Extract worker-models (4 hours)
```bash
# Copy all model modules
cp -r bin/worker-orcd/src/models/* bin/worker-crates/worker-models/src/

# Add worker-gguf dependency
# Update imports
# Test
cargo test -p worker-models
```

**Estimated**: 4 hours  
**Complexity**: Medium (~800 lines, depends on worker-gguf)

### 4. Extract worker-common (3 hours)
```bash
# Copy common modules
cp bin/worker-orcd/src/error.rs bin/worker-crates/worker-common/src/
cp bin/worker-orcd/src/sampling_config.rs bin/worker-crates/worker-common/src/sampling.rs
cp bin/worker-orcd/src/inference_result.rs bin/worker-crates/worker-common/src/inference.rs
cp bin/worker-orcd/src/startup.rs bin/worker-crates/worker-common/src/callback.rs

# Update lib.rs
# Test
cargo test -p worker-common
```

**Estimated**: 3 hours  
**Complexity**: Low (straightforward copy)

### 5. Extract worker-http (4 hours)
```bash
# Copy HTTP modules
cp -r bin/worker-orcd/src/http/* bin/worker-crates/worker-http/src/

# Update dependencies
# Update imports
# Test
cargo test -p worker-http
```

**Estimated**: 4 hours  
**Complexity**: Medium (depends on worker-common)

### 6. Refactor worker-orcd (1 day)
```bash
# Update Cargo.toml to use shared crates
# Update imports throughout worker-orcd
# Implement ComputeBackend for CUDA
# Verify all tests pass
cargo test -p worker-orcd
```

**Estimated**: 1 day  
**Complexity**: High (touches entire codebase)

---

## Timeline Summary

| Phase | Task | Duration | Complexity |
|-------|------|----------|------------|
| 1.1 | Extract worker-gguf | 2 hours | Low |
| 1.2 | Extract worker-tokenizer | 6 hours | Medium |
| 1.3 | Extract worker-models | 4 hours | Medium |
| 1.4 | Extract worker-common | 3 hours | Low |
| 1.5 | Extract worker-http | 4 hours | Medium |
| 2.0 | Refactor worker-orcd | 1 day | High |
| **TOTAL** | **Phase 1-2** | **1-2 days** | - |

---

## Success Criteria

### Phase 1 (Extraction)
- [x] All 6 worker crates scaffold created
- [x] All crates compile independently
- [x] Root Cargo.toml updated
- [x] README.md created
- [ ] worker-gguf extracted and tested
- [ ] worker-tokenizer extracted and tested
- [ ] worker-models extracted and tested
- [ ] worker-common extracted and tested
- [ ] worker-http extracted and tested

### Phase 2 (Refactor)
- [ ] worker-orcd uses shared crates
- [ ] CudaBackend implements ComputeBackend
- [ ] All worker-orcd tests pass
- [ ] No performance regression
- [ ] Binary size similar to before

---

## References

- **Development Plan**: `.docs/WORKER_AARMD_DEVELOPMENT_PLAN.md`
- **Worker Crates README**: `bin/worker-crates/README.md`
- **System Spec**: `bin/.specs/00_llama-orch.md` (SYS-6.3.x)
- **Worker-orcd Spec**: `bin/.specs/01_M0_worker_orcd.md`
- **Root Cargo.toml**: `Cargo.toml` (lines 107-118)

---

## Team AARM Next Actions

1. **Review this scaffold** — Verify structure and dependencies
2. **Begin Phase 1.1** — Extract worker-gguf (2 hours, easiest)
3. **Continue sequentially** — Extract remaining crates
4. **Daily standup** — Track progress, blockers
5. **Phase 2 gate** — All extractions complete before refactoring worker-orcd

---

**Scaffold Complete**: ✅  
**Ready for Extraction**: ✅  
**Estimated Time to Phase 2**: 1-2 days
