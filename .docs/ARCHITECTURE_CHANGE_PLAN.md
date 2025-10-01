# Architecture Change Plan: Custom Worker Implementation

**Date**: 2025-10-01  
**Status**: Planning  
**Impact**: Major architectural refactor

---

## Executive Summary

Management has decided to **ditch external engines** (llama.cpp, vLLM, TGI, Triton) and build a **custom worker** in-house. This requires removing:

1. **Engine provisioner** (`libs/provisioners/engine-provisioner/`)
2. **All worker adapters** (`libs/worker-adapters/`)
3. **Adapter host** (`libs/adapter-host/`)
4. **Engine catalog** (references in specs)

And implementing:

- Custom worker with native inference capabilities
- Direct orchestration without adapter layer
- Simplified pool management

---

## Current Architecture Analysis

### Components to Remove

#### 1. Engine Provisioner (`libs/provisioners/engine-provisioner/`)

**Purpose**: Downloads, compiles, and starts external engines  
**Dependencies**:

- Used by `pool-managerd` in `src/lifecycle/preload.rs`
- Referenced in specs: `.specs/50-engine-provisioner.md`
- Has BDD subcrate: `libs/provisioners/engine-provisioner/bdd/`

**Files to delete**:

```
libs/provisioners/engine-provisioner/
  ├── src/
  ├── bdd/
  ├── .specs/
  ├── Cargo.toml
  └── README.md
```

#### 2. Worker Adapters (`libs/worker-adapters/`)

**Purpose**: Translate orchestrator requests to engine-native APIs  
**Adapters**:

- `llamacpp-http/` - llama.cpp HTTP adapter
- `vllm-http/` - vLLM adapter
- `tgi-http/` - Text Generation Inference adapter  
- `triton/` - Triton adapter
- `openai-http/` - OpenAI-compatible adapter
- `mock/` - Mock adapter (used in tests - may keep for testing)
- `http-util/` - Shared HTTP utilities
- `adapter-api/` - Common trait definition

**Dependencies**:

- Used by `orchestratord` via `adapter-host`
- Referenced in 13 test files
- Specs: `.specs/35-worker-adapters.md`, `.specs/40-44-*.md`

**Files to delete**:

```
libs/worker-adapters/
  ├── llamacpp-http/
  ├── vllm-http/
  ├── tgi-http/
  ├── triton/
  ├── openai-http/
  ├── http-util/
  ├── adapter-api/
  ├── .specs/
  └── README.md
```

**Note**: Consider keeping `mock/` adapter for testing purposes if needed.

#### 3. Adapter Host (`libs/adapter-host/`)

**Purpose**: Adapter registry and dispatch facade  
**Dependencies**:

- Used by `orchestratord` in `src/state.rs`
- Manages adapter lifecycle and routing

**Files to delete**:

```
libs/adapter-host/
  ├── src/
  ├── .specs/
  ├── Cargo.toml
  └── README.md
```

#### 4. Model Provisioner (`libs/provisioners/model-provisioner/`)

**Purpose**: Resolves model references and manages catalog  
**Decision**: **EVALUATE** - May repurpose for custom worker's model loading

**Options**:

- A) Keep and adapt for custom worker model management
- B) Remove and implement simpler model loading in custom worker
- C) Merge relevant functionality into catalog-core

#### 5. Engine Catalog

**Purpose**: Tracks available engines across nodes  
**Decision**: Remove from specs (`.specs/56-engine-catalog.md`)

---

## Impact Analysis

### Direct Code Dependencies

#### `bin/orchestratord/`

**Files affected**:

- `src/state.rs` - Uses `AdapterHost` (line 6, 21, 73)
- `src/app/bootstrap.rs` - Registers adapters
- `src/services/streaming.rs` - Dispatches via adapter-host
- `Cargo.toml` - Dependencies on adapter-host and worker-adapters

**Required changes**:

- Remove `adapter_host: Arc<AdapterHost>` from `AppState`
- Replace adapter dispatch with direct worker communication
- Update streaming service to use new worker protocol

#### `bin/pool-managerd/`

**Files affected**:

- `src/lifecycle/preload.rs` - Uses `provisioners_engine_provisioner::PreparedEngine` (line 24)
- `src/core/registry.rs` - References engine provisioner
- `src/validation/preflight.rs` - Validates engine tooling
- `Cargo.toml` - Dependency on engine-provisioner

**Required changes**:

- Remove engine provisioning logic
- Implement custom worker lifecycle management
- Update registry to track custom workers instead of external engines

#### `Cargo.toml` (workspace)

**Members to remove**:

```toml
"libs/worker-adapters/adapter-api",
"libs/worker-adapters/llamacpp-http",
"libs/worker-adapters/vllm-http",
"libs/worker-adapters/tgi-http",
"libs/worker-adapters/triton",
"libs/worker-adapters/mock",
"libs/worker-adapters/openai-http",
"libs/worker-adapters/http-util",
"libs/worker-adapters/http-util/bdd",
"libs/adapter-host",
"libs/provisioners/engine-provisioner",
"libs/provisioners/engine-provisioner/bdd",
# Evaluate:
"libs/provisioners/model-provisioner",
"libs/provisioners/model-provisioner/bdd",
```

### Test Dependencies

**BDD Tests**:

- `test-harness/bdd/` - Uses mock adapter, needs update
- `bin/orchestratord/bdd/` - Tests adapter dispatch
- `bin/pool-managerd/bdd/` - Tests engine provisioning

**Integration Tests**:

- Multiple tests reference worker adapters
- Determinism suite may test adapter behavior
- E2E tests use adapter layer

**Action**: Update or remove tests that depend on removed components.

### Specification Dependencies

**Specs to update/remove**:

- `.specs/35-worker-adapters.md` - Root worker adapters spec
- `.specs/40-worker-adapters-llamacpp-http.md`
- `.specs/41-worker-adapters-vllm-http.md`
- `.specs/42-worker-adapters-tgi-http.md`
- `.specs/43-worker-adapters-triton.md`
- `.specs/44-worker-adapters-openai-http.md`
- `.specs/50-engine-provisioner.md`
- `.specs/56-engine-catalog.md`
- `.specs/00_llama-orch.md` - Update §2.12 (Engine Provisioning) and §4 (Engines & Adapters)

### Documentation Dependencies

**Files to update**:

- `README.md` - Architecture diagrams, binaries map, workspace map
- `bin/orchestratord/README.md` - Remove adapter references
- `bin/pool-managerd/README.md` - Remove engine provisioning references
- `docs/CONFIGURATION.md` - Remove engine/adapter config
- `COMPLIANCE.md` - Remove adapter requirement IDs
- Architecture diagrams (Mermaid) in README

---

## New Architecture Proposal

### Custom Worker Design

**Location**: `libs/worker/` or `bin/workerd/`

**Responsibilities**:

1. **Native inference** - Direct LLM execution without external engines
2. **Model loading** - Load and manage model weights
3. **GPU management** - Direct CUDA/GPU interaction
4. **Token generation** - Streaming token generation
5. **Session management** - KV cache management

**Options**:

#### Option A: Library + Integration

- Create `libs/worker/` library with worker logic
- Integrate directly into pool-managerd
- pool-managerd manages worker lifecycle

#### Option B: Separate Binary

- Create `bin/workerd/` as standalone worker binary
- pool-managerd spawns and manages worker processes
- Similar to current engine model but internal

#### Option C: Embedded in pool-managerd

- Implement worker directly in pool-managerd
- Simpler architecture, tighter coupling
- No separate process management

**Recommendation**: Start with **Option C** for MVP, migrate to Option A/B if needed.

### Communication Flow

**Before** (with adapters):

```
Client → orchestratord → adapter-host → worker-adapter → external-engine → GPU
```

**After** (direct):

```
Client → orchestratord → pool-managerd → custom-worker → GPU
```

**Or even simpler**:

```
Client → orchestratord → GPU (via embedded worker)
```

### Required Components

1. **Model loader** - GGUF parsing or safetensors loading
2. **Tokenizer** - Token encoding/decoding
3. **Inference engine** - Forward pass, sampling, generation
4. **CUDA integration** - GPU memory management, kernel execution
5. **KV cache manager** - Session state management
6. **Streaming protocol** - SSE token emission

### Technology Choices

**Inference Backend Options**:

- **candle-rs** - Rust ML framework with CUDA support
- **burn** - Rust deep learning framework
- **tch-rs** - Rust bindings for PyTorch
- **ggml-rs** - Rust bindings for GGML (llama.cpp core)
- **Custom implementation** - Build from scratch (high effort)

**Recommendation**: Evaluate **candle-rs** first (pure Rust, good CUDA support).

---

## Migration Strategy

### Phase 1: Preparation (Current)

- [x] Survey codebase for dependencies
- [ ] Create detailed removal plan (this document)
- [ ] Archive current state to branch
- [ ] Create feature branch for changes

### Phase 2: Cleanup

1. **Remove worker adapters**
   - Delete `libs/worker-adapters/` (keep mock if needed for tests)
   - Remove from workspace Cargo.toml
   - Update affected tests

2. **Remove adapter-host**
   - Delete `libs/adapter-host/`
   - Remove from workspace Cargo.toml
   - Update orchestratord to remove adapter dependencies

3. **Remove engine-provisioner**
   - Delete `libs/provisioners/engine-provisioner/`
   - Remove from workspace Cargo.toml
   - Update pool-managerd preload logic

4. **Evaluate model-provisioner**
   - Decide: keep, remove, or repurpose
   - If keeping, adapt for custom worker

5. **Update specifications**
   - Remove/archive adapter specs
   - Remove engine provisioner spec
   - Update core spec (`.specs/00_llama-orch.md`)

6. **Update documentation**
   - Update README.md architecture diagrams
   - Update component READMEs
   - Remove adapter references

### Phase 3: Custom Worker Implementation

1. **Design custom worker architecture**
   - Choose inference backend (candle-rs/burn/tch-rs)
   - Design model loading and GPU management
   - Design streaming protocol

2. **Implement MVP worker**
   - Basic model loading
   - Simple inference with one model
   - Token streaming
   - CUDA integration

3. **Integrate with orchestratord**
   - Direct worker dispatch
   - Update streaming service
   - Update admission/placement logic

4. **Integrate with pool-managerd**
   - Worker lifecycle management
   - Health monitoring
   - Readiness tracking

### Phase 4: Testing & Validation

1. **Update test suite**
   - Remove adapter tests
   - Add worker tests
   - Update BDD scenarios

2. **Run full test suite**
   - Unit tests
   - Integration tests
   - BDD tests
   - Determinism suite

3. **Performance validation**
   - Benchmark against previous architecture
   - GPU utilization
   - Latency/throughput

### Phase 5: Documentation & Cleanup

1. **Update all documentation**
2. **Clean up dead code**
3. **Archive removed components**
4. **Update COMPLIANCE.md**

---

## Risk Assessment

### High Risk

- **Breaking existing functionality** - Major architectural change
- **Performance regression** - Custom worker may be slower initially
- **CUDA integration complexity** - Direct GPU management is non-trivial

### Medium Risk

- **Test coverage gaps** - Removing adapters removes test coverage
- **Model compatibility** - Custom worker may support fewer models
- **Development timeline** - Building custom worker takes time

### Low Risk

- **Reversibility** - Changes can be backed out if needed
- **Incremental migration** - Can keep mock adapter for testing

---

## Decision Points

### Immediate Decisions Needed

1. **Model Provisioner**: Keep, remove, or repurpose?
   - **Recommendation**: Keep and adapt for custom worker

2. **Mock Adapter**: Keep for testing?
   - **Recommendation**: Keep temporarily, remove after worker is stable

3. **Worker Architecture**: Library, binary, or embedded?
   - **Recommendation**: Start embedded (Option C), extract later if needed

4. **Inference Backend**: Which Rust ML framework?
   - **Recommendation**: Evaluate candle-rs first

5. **Migration Approach**: Big bang or incremental?
   - **Recommendation**: Incremental - remove components first, implement worker second

---

## Timeline Estimate

**Phase 1 (Preparation)**: 1 day

- Complete removal plan ✓
- Archive and branch

**Phase 2 (Cleanup)**: 2-3 days

- Remove adapters and engine-provisioner
- Update code dependencies
- Update documentation

**Phase 3 (Custom Worker)**: 1-2 weeks

- Design architecture: 2 days
- Implement MVP: 5-7 days
- Integration: 2-3 days

**Phase 4 (Testing)**: 3-5 days

- Update tests
- Run validation
- Fix issues

**Phase 5 (Documentation)**: 2 days

- Final documentation
- Cleanup

**Total**: 3-4 weeks

---

## Next Steps

1. **Get approval** from management on:
   - Architecture approach (Option A/B/C)
   - Inference backend choice
   - Timeline

2. **Create feature branch**: `feat/custom-worker-refactor`

3. **Start Phase 2**: Remove external engine dependencies

4. **Parallel track**: Research and prototype custom worker

---

## Questions for Management

1. What inference capabilities are required in the custom worker?
   - Models to support (size, architecture)?
   - Inference modes (completion, chat, embeddings)?
   - Performance targets (tokens/sec)?

2. Do we need to maintain backward compatibility?
   - API compatibility?
   - Configuration format?
   - Client SDK?

3. What's the priority: speed vs. feature completeness?
   - MVP with one model type?
   - Full multi-model support from day one?

4. Are there any existing inference codebases to leverage?
   - Internal libraries?
   - Preferred frameworks?

5. Timeline constraints?
   - Hard deadline?
   - Phased rollout acceptable?

---

**Status**: Awaiting decisions on architecture approach and inference backend before proceeding with cleanup phase.
