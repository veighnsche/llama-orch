# Architecture Change Plan: Custom Worker Implementation

**Date**: 2025-10-01  
**Status**: Planning  
**Impact**: Major architectural refactor

---

## Executive Summary

Strategic decision to **drop external engine support** (llama.cpp, vLLM, TGI, Triton) and build a **custom in-process worker** (`worker-orcd`).

### Why

The control we need—exact VRAM residency, deterministic sharding for tensor-parallel models, strict "no RAM inference" policy—is not achievable reliably by wrapping external engines. Wrapping forces us to fight opaque allocators, hidden uploads, and per-engine semantics.

Building in-process gives us clean guarantees: **no mixed VRAM/RAM runtime**, **shard-level TP control**, **atomic seeding/eviction policies**, **audited staging**, and **deterministic performance for SLOs**.

### New Architecture: Three Binaries

```
orchestord       → Orchestrator (no GPU required)
pool-managerd    → Pool manager (staging, catalog, policy, eviction)
worker-orcd      → Worker (one process per GPU, accepts sealed VRAM shards)
```

### Key Technical Changes

**Components to remove**:

1. **Engine provisioner** (`libs/provisioners/engine-provisioner/`)
2. **All worker adapters** (`libs/worker-adapters/`)
3. **Adapter host** (`libs/adapter-host/`)
4. **Engine catalog** (references in specs)

And implementing:

- Custom worker (`worker-orcd`) with native inference capabilities
- Direct orchestration without adapter layer
- Simplified pool management with sealed VRAM guarantees

---

## Current Architecture Analysis

### Components to Remove

#### 1. Engine Provisioner (`libs/provisioners/engine-provisioner/`)

/// REMOVAL APPROVED

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
  ├── tests/  (already removed by user)
  ├── Cargo.toml
  └── README.md
```

#### 2. Worker Adapters (`libs/worker-adapters/`)

/// REMOVAL APPROVED, we're going to have to deal with the fact that different models need different engines. We MIGHT need an adapter for that. But this SHOULD be a problem for the worker-orcd, so the wo

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

## New Architecture: worker-orcd

### Design Philosophy

**Core principle**: Workers accept **sealed VRAM shards** with attestation. No surprises, no implicit uploads, no black-box paging rules. Policy checks happen before any GPU is instructed to run.

### Three-Binary System

```
orchestratord       → Orchestrator (no GPU required)
                      - Task admission, queueing, placement
                      - SSE streaming to clients
                      - Multi-node coordination

pool-managerd       → Pool manager (per host)
                      - Model staging and catalog
                      - Policy enforcement (VRAM-only, eviction rules)
                      - Worker lifecycle management
                      - Audited staging before GPU execution

worker-orcd         → Worker (one process per GPU)
                      - Accepts ModelShardHandles (sealed VRAM shards)
                      - Native inference execution
                      - KV cache management
                      - NCCL coordination for tensor-parallel
```

### Communication Flow

**Before** (with external engines):

```
Client → orchestratord → adapter-host → worker-adapter → llama.cpp/vLLM → GPU
                                                          (opaque allocator)
```

**After** (direct control):

```
Client → orchestratord → pool-managerd → worker-orcd → GPU
                         (audited staging) (sealed shards)
```

### ModelShardHandle Contract

**Core abstraction**: A `ModelShardHandle` is a guarantee that model weights are resident in VRAM and cannot be evicted without explicit coordination.

```rust
pub struct ModelShardHandle {
    pub shard_id: String,           // Unique shard identifier
    pub gpu_device: u32,            // CUDA device index
    pub vram_ptr: *mut c_void,      // VRAM pointer (opaque to caller)
    pub vram_bytes: usize,          // Size in VRAM
    pub sealed: bool,               // Attestation: true = resident, immutable
    pub digest: String,             // SHA256 of shard data (determinism)
    pub model_ref: String,          // Original model reference
    pub shard_index: Option<usize>, // For tensor-parallel: which shard
    pub total_shards: Option<usize>,// For tensor-parallel: total count
}
```

**Sealed guarantee**: When `sealed == true`, the shard:

- Is resident in VRAM at `vram_ptr`
- Cannot be paged out or moved without explicit eviction API call
- Has verified digest matching `digest` field
- Is ready for immediate inference

### Worker RPC Protocol (Plan/Commit/Ready/Execute)

**1. Plan** — Determine feasibility

```
POST /worker/plan
{
  "model_ref": "hf:meta-llama/Llama-3.1-8B",
  "shard_layout": "single" | "tensor_parallel",
  "tp_degree": 2  // if tensor_parallel
}

Response:
{
  "feasible": bool,
  "vram_required": usize,
  "shard_plan": [
    { "shard_index": 0, "vram_bytes": ..., "gpu_device": 0 },
    { "shard_index": 1, "vram_bytes": ..., "gpu_device": 1 }
  ]
}
```

**2. Commit** — Load model into VRAM and seal

```
POST /worker/commit
{
  "model_ref": "hf:meta-llama/Llama-3.1-8B",
  "shard_id": "shard-0",
  "shard_index": 0,
  "model_bytes": <binary or path>
}

Response:
{
  "handle": ModelShardHandle,
  "sealed": true
}
```

**3. Ready** — Attest that worker is ready with sealed shards

```
GET /worker/ready

Response:
{
  "ready": bool,
  "handles": [ModelShardHandle],
  "nccl_group_id": "..." // if TP
}
```

**4. Execute** — Run inference with sealed shard

```
POST /worker/execute
{
  "handle_id": "shard-0",
  "prompt": "Hello, world!",
  "params": {
    "max_tokens": 100,
    "temperature": 0.7,
    "seed": 42
  }
}

Response: SSE stream
event: token
data: {"t": "Hello", "i": 0}

event: token
data: {"t": "!", "i": 1}

event: end
data: {"tokens_out": 2, "decode_time_ms": 45}
```

### Tensor-Parallel Support (NCCL)

**Multi-GPU sharding**: When a model is too large for one GPU, `pool-managerd` plans a shard layout and coordinates multiple `worker-orcd` processes:

1. **Shard planning** (pool-managerd):
   - Divide model into N shards (equal or weighted by layer)
   - Assign each shard to a GPU device
   - Generate `ModelShardHandle` for each

2. **NCCL group setup** (worker-orcd):
   - Workers join NCCL communicator group
   - Exchange `nccl_group_id` for coordination
   - All-reduce operations for cross-GPU computation

3. **Inference execution**:
   - Each worker runs forward pass on its shard
   - NCCL all-reduce for attention/FFN cross-GPU ops
   - Final worker emits tokens to orchestratord

**Example**: Llama-70B on 4x RTX 3090 (24GB each)

- Shard 0: Layers 0-19 → GPU 0
- Shard 1: Layers 20-39 → GPU 1
- Shard 2: Layers 40-59 → GPU 2
- Shard 3: Layers 60-79 → GPU 3

### M0 Pilot Scope

**Goal**: Prove the sealed VRAM shard concept with minimal inference.

**In scope**:

- cuBLAS for matrix multiplication
- Naive attention (non-fused, simple QKV matmul)
- Simple sampling (greedy or top-k)
- Single-GPU only (no TP)
- GGUF model loading
- Token streaming via SSE

**Out of scope (post-M0)**:

- Continuous batching
- Fused attention kernels (FlashAttention)
- Tensor-parallel (multi-GPU)
- vLLM-class throughput optimizations
- Multiple model formats beyond GGUF

**Success criteria**:

1. Load model into VRAM via `Commit` API
2. Attest sealed status via `Ready` API
3. Run greedy inference and stream tokens
4. Verify determinism (same seed → same tokens)
5. Validate VRAM-only policy (no RAM inference)

### Language & Tooling Stack

**Architecture decision**: Hybrid Rust + CUDA C++ approach

**Rust for control plane**:

- Job lifecycle management
- VRAM residency enforcement (safety guarantees)
- Telemetry and observability
- Scheduling hooks
- RPC server (Plan/Commit/Ready/Execute endpoints)
- Consistency with other daemons (orchestord, pool-managerd)

**CUDA C++ for compute kernels**:

- cuBLAS/CUTLASS for dense matrix operations
- NCCL for multi-GPU coordination
- Custom attention kernels (prefill + decode)
- Quantization kernels (Q4_0, Q5_1, etc.)
- RoPE and other positional encodings

**Rust ↔ CUDA FFI boundary**:

- Thin, explicit interface layer
- `cudarc` or `cust` for safe CUDA bindings
- Rust drives control logic, CUDA handles heavy math
- Clear ownership of VRAM pointers

**Rationale**:

- **Safety**: Rust ensures VRAM-only inference and strict residency enforcement
- **Performance**: CUDA libraries provide peak throughput with custom kernel flexibility
- **Transparency**: Operators see exactly what's happening at each layer
- **Consistency**: All daemons share Rust as a base language

### Model Compatibility System (MCD/ECP)

**Problem**: Different models require different capabilities (attention mechanisms, RoPE variants, quantization formats). Instead of "one engine per model," we implement explicit capability matching.

**Model Capability Descriptor (MCD)**: Metadata embedded in each model artifact

```json
{
  "model_id": "meta-llama/Llama-3.1-8B",
  "positional": "rope_llama",
  "attention": "gqa",
  "quant": ["q4_0", "q8_0"],
  "context_max": 8192,
  "vocab_size": 128256
}
```

**Engine Capability Profile (ECP)**: What a given worker-orcd supports

```json
{
  "worker_id": "worker-orcd-gpu0",
  "supports_positional": ["rope_llama", "rope_neox", "alibi"],
  "supports_attention": ["mha", "gqa", "mqa"],
  "supports_quant": ["q4_0", "q5_1", "q8_0"],
  "max_context": 16384,
  "vram_bytes": 24000000000
}
```

**Matching logic**: Orchestrator assigns jobs only if `MCD ⊆ ECP`

- If model requires `rope_llama`, worker must support `rope_llama`
- If model requires `gqa`, worker must support `gqa`
- If model uses `q4_0`, worker must support `q4_0`
- Compatibility is explicit and deterministic

**Benefits**:

- **Transparency**: Operators see exactly which models are runnable where
- **No silent failures**: Mismatches caught at admission time
- **Flexibility**: Add new capabilities without breaking existing models
- **Extensibility**: Easy to add new attention variants, quant formats, etc.

---

## Migration Strategy

### Phase 1: Preparation (Current)

- [x] Survey codebase for dependencies
- [x] Create detailed removal plan (this document)
- [x] Management proposal submitted
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
   - Decision: Repurpose for `worker-orcd` model staging
   - Adapt for sealed VRAM shard preparation

5. **Update specifications**
   - Remove/archive adapter specs
   - Remove engine provisioner spec
   - Update core spec (`.specs/00_llama-orch.md`)
   - Create new `worker-orcd` spec

6. **Update documentation**
   - Update README.md architecture diagrams
   - Update component READMEs
   - Remove adapter references

### Phase 3: worker-orcd M0 Pilot

#### Task Group 1: Rust Control Layer (2-3 days)

**Objective**: Implement worker control plane in Rust

**Sub-tasks**:

1. **Crate structure setup**
   - Create `bin/worker-orcd/` crate
   - Set up directory structure: `src/`, `cuda/`, `tests/`, `.specs/`
   - Add dependencies: `axum`, `tokio`, `serde`, `cudarc`, `tracing`

2. **Job lifecycle module** (`src/lifecycle.rs`)
   - Define `Job` struct with `job_id`, `model_ref`, `params`
   - Implement state machine: `Pending → Loading → Executing → Completed`
   - Add lifecycle hooks for telemetry

3. **VRAM residency enforcement** (`src/residency.rs`)
   - Implement `ModelShardHandle` struct (from plan)
   - Add `seal_shard()` and `validate_sealed()` functions
   - VRAM-only attestation: verify no host memory used during inference
   - Add runtime checks that fail fast if RAM inference detected

4. **Telemetry module** (`src/telemetry.rs`)
   - Structured logging with `tracing`
   - Metrics: VRAM usage, job latency, token throughput
   - Narration support per spec (ORCH-33xx)

5. **Scheduling hooks** (`src/scheduler.rs`)
   - Single-slot scheduler for M0 (one job at a time)
   - Queue interface (deferred to orchestratord for M0)

6. **RPC server** (`src/server.rs`)
   - Implement Plan endpoint: check VRAM feasibility
   - Implement Commit endpoint: load model, return sealed handle
   - Implement Ready endpoint: attest worker status
   - Implement Execute endpoint: run inference, stream tokens

#### Task Group 2: CUDA FFI Boundary (2-3 days)

**Objective**: Define clean Rust ↔ CUDA interface

**Sub-tasks**:

1. **CUDA headers** (`cuda/kernels.h`)
   - Define C-compatible structs for model metadata
   - Declare function signatures for GEMM, RoPE, attention
   - Add error handling enums

2. **FFI wrapper** (`src/cuda_ffi.rs`)
   - Expose Rust-safe wrappers via `cudarc` or `cust`
   - Implement `CudaContext` for device management
   - Add VRAM allocation/deallocation helpers
   - Implement pointer safety checks

3. **cuBLAS integration** (`src/cuda_ffi.rs`)
   - Initialize cuBLAS handle
   - Wrap `cublasSgemm` for single-precision matmul
   - Add batched GEMM support for multi-sequence

4. **Error handling**
   - Map CUDA errors to Rust `Result` types
   - Add detailed error messages for debugging
   - Implement fail-fast on CUDA/driver errors

#### Task Group 3: Initial Kernel Set (3-5 days)

**Objective**: Implement core CUDA kernels for inference

**Sub-tasks**:

1. **cuBLAS GEMM integration** (`cuda/gemm.cu`)
   - Wrap cuBLAS `Sgemm` for forward pass
   - Optimize for row-major layout (Llama/Transformer format)
   - Benchmark: compare against naive matmul

2. **RoPE kernel** (`cuda/rope.cu`)
   - Implement RoPE (Rotary Position Embedding) for Llama
   - Support `rope_llama` variant (freq_base=10000)
   - Add `rope_neox` variant for compatibility
   - Test: verify against reference implementation

3. **Attention kernel** (`cuda/attention.cu`)
   - Implement prefill attention (full Q·K^T, softmax, ·V)
   - Implement decode attention (single query, cached K/V)
   - Use naive attention (no FlashAttention fusion for M0)
   - Support GQA (Grouped Query Attention) with configurable groups
   - Test: verify against PyTorch reference

4. **RMSNorm kernel** (`cuda/rmsnorm.cu`)
   - Implement RMSNorm (used in Llama pre/post layer)
   - Fuse with weight multiplication where possible

5. **Sampling kernel** (`cuda/sampling.cu`)
   - Implement greedy sampling (argmax)
   - Add top-k sampling (optional for M0)
   - Add temperature scaling

#### Task Group 4: Model Loading & Execution (2-3 days)

**Objective**: Load models and run inference end-to-end

**Sub-tasks**:

1. **GGUF model loader** (`src/loader/gguf.rs`)
   - Parse GGUF file format (metadata + weights)
   - Load weights directly into VRAM
   - Validate model signature/digest
   - Support Q4_0 quantization format (dequantize on load for M0)

2. **Inference engine** (`src/engine.rs`)
   - Implement forward pass: RMSNorm → Attention → FFN → RMSNorm
   - Call CUDA kernels via FFI boundary
   - Manage KV cache in VRAM
   - Generate tokens autoregressively

3. **Token streaming** (`src/streaming.rs`)
   - Implement SSE streaming for Execute endpoint
   - Emit `event: token` with `{"t": "...", "i": ...}`
   - Emit `event: end` with metadata (tokens_out, decode_time_ms)

#### Task Group 5: MCD/ECP Schema (1-2 days)

**Objective**: Define and implement capability matching

**Sub-tasks**:

1. **MCD schema** (`contracts/mcd-schema.json`)
   - Define JSON schema for Model Capability Descriptor
   - Fields: `positional`, `attention`, `quant`, `context_max`, `vocab_size`
   - Add validation rules

2. **ECP schema** (`contracts/ecp-schema.json`)
   - Define JSON schema for Engine Capability Profile
   - Fields: `supports_positional`, `supports_attention`, `supports_quant`, `max_context`, `vram_bytes`
   - Add validation rules

3. **Capability matching logic** (`src/capabilities.rs`)
   - Implement `MCD ⊆ ECP` checker
   - Parse MCD from model metadata (embedded in GGUF or sidecar JSON)
   - Parse ECP from worker config or runtime introspection
   - Return compatibility verdict: `Compatible | Incompatible(reason)`

4. **Integration with Plan endpoint**
   - Plan endpoint checks MCD ⊆ ECP before returning `feasible: true`
   - Return detailed error if incompatible: `"Model requires rope_llama, worker only supports rope_neox"`

#### Task Group 6: Integration with pool-managerd (2-3 days)

**Objective**: Connect worker-orcd to pool manager lifecycle

**Sub-tasks**:

1. **Worker spawn logic** (`bin/pool-managerd/src/lifecycle/worker.rs`)
   - Replace engine-provisioner calls with worker-orcd spawn
   - Pass GPU device index, config, model path
   - Monitor worker process health

2. **Health monitoring**
   - pool-managerd polls Ready endpoint
   - Update registry when worker reports ready
   - Mark worker unready on timeout/error

3. **Sealed shard handoff**
   - pool-managerd stages model in host memory
   - Calls Commit endpoint to load into VRAM
   - Validates sealed=true in response
   - Updates registry with sealed handle metadata

4. **Policy enforcement**
   - pool-managerd validates VRAM-only policy before worker spawn
   - Rejects models that exceed VRAM capacity
   - Enforces eviction rules when needed

#### Task Group 7: Validation & Testing (1-2 days)

**Objective**: Prove M0 pilot works end-to-end

**Sub-tasks**:

1. **Load TinyLlama-1.1B**
   - Download GGUF file
   - Load via Commit endpoint
   - Verify sealed status

2. **Run inference**
   - Execute with prompt: "Once upon a time"
   - Stream tokens via SSE
   - Verify output is coherent (basic sanity check)

3. **Determinism test**
   - Run same prompt with seed=42 multiple times
   - Verify token streams are identical
   - Document any non-determinism sources

4. **VRAM-only validation**
   - Monitor host memory usage during inference
   - Verify no RAM allocated for model weights or activations
   - Only allow RAM for staging/catalog (not live decode)

**Post-M0 (Phase 3b - optional)**:

- NCCL integration for tensor-parallel
- Multi-GPU shard coordination
- Test Llama-70B on 4x GPU setup
- Quantization support (Q4_0, Q5_1, Q8_0)

### Phase 4: Testing & Validation

1. **Update test suite**
   - Remove adapter tests
   - Add worker-orcd tests
   - Update BDD scenarios
   - Validate determinism

2. **Run full test suite**
   - Unit tests
   - Integration tests
   - BDD tests
   - Determinism suite

3. **Performance validation**
   - Benchmark against previous architecture
   - GPU utilization
   - Latency/throughput
   - VRAM efficiency

### Phase 5: Production Hardening

1. **Optimize inference**
   - Add continuous batching
   - Optimize kernel usage
   - Memory pooling
   - Prefetching

2. **Documentation & Cleanup**
   - Complete API documentation
   - Deployment guides
   - Clean up dead code
   - Update COMPLIANCE.md

---

## Risk Assessment

### High Risk

- **Breaking existing functionality** - Major architectural change
- **Performance regression** - Custom worker may be slower initially
- **CUDA integration complexity** - Direct GPU management is non-trivial
- **Tensor-parallel correctness** - Multi-GPU sharding is complex

### Medium Risk

- **Test coverage gaps** - Removing adapters removes test coverage
- **Model compatibility** - Custom worker may support fewer models initially
- **Development timeline** - Building custom worker takes time
- **NCCL debugging** - Multi-GPU coordination can be difficult to debug

### Low Risk

- **Reversibility** - Changes can be backed out if needed (keep branch)
- **Incremental migration** - Can keep mock adapter for testing
- **Customer opt-out** - Can offer adapter compatibility layer

---

## Decision Points

### Key Decisions (Locked In)

1. **Worker Architecture**: `bin/worker-orcd/` - one process per GPU
   - Accepts sealed VRAM shards via ModelShardHandle
   - Plan/Commit/Ready/Execute RPC protocol

2. **Inference Backend**: cuBLAS + naive attention for M0
   - Simple, auditable, deterministic
   - Optimize post-M0 (FlashAttention, batching, etc.)

3. **Tensor-Parallel Strategy**: NCCL-based sharding (post-M0)
   - pool-managerd plans shard layout
   - worker-orcd processes coordinate via NCCL

4. **Model Provisioner**: Repurpose for worker-orcd staging
   - Adapt for sealed VRAM shard preparation
   - Keep catalog integration

5. **Migration Approach**: Incremental
   - Phase 2: Cleanup (remove adapters)
   - Phase 3: M0 pilot (prove concept)
   - Phase 4: Production hardening

### Open Questions

1. **Mock adapter during transition**: Keep temporarily for testing?
   - **Recommendation**: Keep until worker-orcd is stable

2. **Adapter compatibility layer**: Offer for existing customers?
   - **Recommendation**: Optional, low priority

3. **Timeline constraints**: Hard deadlines?
   - **TBD**: Management input needed

---

## Timeline Estimate

**Phase 1 (Preparation)**: 1 day

- Complete removal plan ✓
- Management proposal ✓
- Archive and branch

**Phase 2 (Cleanup)**: 2-3 days

- Remove adapters and engine-provisioner
- Update code dependencies
- Update documentation

**Phase 3 (M0 Pilot)**: 2-3 weeks

- worker-orcd binary setup: 2 days
- Minimal inference (cuBLAS + naive): 5-7 days
- pool-managerd integration: 2-3 days
- TP sharding validation: 3-5 days

**Phase 4 (Testing)**: 3-5 days

- Update tests
- Run validation
- Fix issues

**Phase 5 (Production)**: 1-2 weeks

- Optimize inference
- Final documentation
- Cleanup

**Total**: 4-6 weeks (includes M0 pilot and production hardening)

---

## Next Steps

1. **Get approval** from management on:
   - Strategic direction (custom worker vs external engines)
   - Three-binary architecture (orchestord / pool-managerd / worker-orcd)
   - ModelShardHandle contract and sealed VRAM guarantees
   - Timeline and resource allocation

2. **Create feature branch**: `feat/custom-worker-orcd`

3. **Start Phase 2**: Remove external engine dependencies

4. **Parallel track**: Implement M0 pilot for worker-orcd

---

## Summary

This plan documents the strategic shift from external engine wrapping to a custom `worker-orcd` implementation that provides:

1. **Sealed VRAM guarantees** via ModelShardHandle contract
2. **Explicit tensor-parallel control** with NCCL coordination (post-M0)
3. **Audited staging** with policy enforcement before GPU execution
4. **Deterministic inference** with no opaque allocators
5. **Three-binary architecture** (orchestord / pool-managerd / worker-orcd)
6. **Hybrid Rust + CUDA C++** for safety and performance
7. **MCD/ECP capability matching** for transparent model compatibility

**Language & Tooling**:

- **Rust**: Control plane, VRAM residency enforcement, telemetry, RPC server
- **CUDA C++**: Compute kernels (cuBLAS, RoPE, attention, sampling)
- **FFI boundary**: Thin interface via `cudarc`, Rust drives logic, CUDA handles math

**Capability System**:

- **MCD** (Model Capability Descriptor): Embedded in model artifacts
- **ECP** (Engine Capability Profile): Advertised by worker-orcd
- **Matching**: Orchestrator assigns jobs only if MCD ⊆ ECP

**M0 Pilot** (7 task groups):

1. Rust control layer (lifecycle, residency, telemetry, RPC)
2. CUDA FFI boundary (cudarc, cuBLAS, error handling)
3. Initial kernels (GEMM, RoPE, attention, RMSNorm, sampling)
4. Model loading & execution (GGUF loader, inference engine, SSE streaming)
5. MCD/ECP schema (JSON schemas, capability matching logic)
6. pool-managerd integration (worker spawn, health, sealed handoff)
7. Validation (TinyLlama, determinism, VRAM-only check)

**Post-M0**: FlashAttention, continuous batching, multi-GPU TP, production optimizations, more quant formats.

**Status**: Ready to begin Phase 2 (cleanup) and Phase 3 (M0 pilot implementation) with clear task breakdown.
