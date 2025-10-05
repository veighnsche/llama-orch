# Worker Adapters SPEC — Pluggable Worker Backend Support (POOL-1xxx)

**Author**: Specs Team  
**Date**: 2025-10-05  
**Status**: Draft (Future Feature - Post M2)  
**Applies to**: `bin/pool-managerd/`  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## 0. Purpose & Motivation

### 0.1 Problem Statement

The current pool-managerd implementation hardcodes worker spawning logic for the bespoke `worker-orcd` binary. This design couples the pool manager to a single worker implementation and prevents extensibility for:

1. **External inference engines** — llama.cpp server, vLLM, TGI (Text Generation Inference)
2. **Specialized hardware** — Apple ARM workers (Metal backend), TPU workers
3. **Different modalities** — Image generation (Stable Diffusion), audio generation, multimodal
4. **Provider-specific optimizations** — Custom kernels, proprietary formats

### 0.2 Solution: Worker Adapter Pattern

Introduce a **worker adapter layer** in pool-managerd that abstracts worker lifecycle and communication protocols. The pool manager orchestrates workers through adapters, enabling support for multiple worker types without changing core orchestration logic.

**Key benefits:**
- ✅ Support multiple inference backends (bespoke, llama.cpp, vLLM, custom)
- ✅ Support multiple modalities (text, image, audio, multimodal)
- ✅ Support specialized hardware (NVIDIA CUDA, Apple Metal, AMD ROCm)
- ✅ Enable provider ecosystem (vendors can supply adapter + worker implementations)
- ✅ Preserve backwards compatibility with existing bespoke worker

### 0.3 Design Principles

1. **Adapters are thin wrappers** — No business logic, only protocol translation
2. **Orchestrator remains agnostic** — Orchestrator sees only worker capabilities, not worker types
3. **Capability-based scheduling** — Orchestrator schedules based on capabilities (text-gen, image-gen), not worker types
4. **Worker protocols are normalized** — All workers expose same HTTP API contract via adapter translation
5. **Configuration over code** — Worker types are registered via config, not hardcoded

---

## 1. Architecture Overview

### 1.1 Current Architecture (Hardcoded)

```
┌─────────────────────────────────────┐
│ Pool Manager                        │
│  ┌──────────────────────────────┐  │
│  │ worker-lifecycle             │  │
│  │  • Hardcoded spawn logic     │  │
│  │  • Hardcoded CLI args        │  │
│  │  • Hardcoded callback format │  │
│  └──────────────────────────────┘  │
└─────────────────┬───────────────────┘
                  │ spawn
                  ↓
         ┌────────────────┐
         │ worker-orcd    │
         │ (bespoke only) │
         └────────────────┘
```

### 1.2 Proposed Architecture (Adapter Pattern)

```
┌──────────────────────────────────────────────────────────┐
│ Pool Manager                                             │
│  ┌───────────────────────────────────────────────────┐  │
│  │ worker-lifecycle (adapter-aware)                  │  │
│  │  • Select adapter by worker_type                  │  │
│  │  • Delegate spawn/stop/health to adapter         │  │
│  │  • Normalize worker state via adapter interface  │  │
│  └────────────────┬──────────────────────────────────┘  │
│                   │ dispatch                             │
│  ┌────────────────┴──────────────────────────────────┐  │
│  │ Worker Adapter Registry                           │  │
│  │  • BespokeCudaAdapter (default)                   │  │
│  │  • LlamaCppAdapter                                │  │
│  │  • AppleMetalAdapter                              │  │
│  │  • ImageGenAdapter (Stable Diffusion)             │  │
│  └───┬───────┬──────────┬─────────────────┬──────────┘  │
└──────┼───────┼──────────┼─────────────────┼──────────────┘
       │       │          │                 │
     spawn   spawn      spawn             spawn
       │       │          │                 │
       ↓       ↓          ↓                 ↓
  ┌────────┐ ┌────────┐ ┌────────┐ ┌──────────────┐
  │worker- │ │llama.  │ │worker- │ │sd-worker     │
  │orcd    │ │cpp     │ │orcd-arm│ │(image-gen)   │
  │(CUDA)  │ │server  │ │(Metal) │ │              │
  └────────┘ └────────┘ └────────┘ └──────────────┘
```

---

## 2. Core Requirements

### [POOL-1001] Adapter Interface

Pool-managerd MUST define a `WorkerAdapter` trait with lifecycle operations:

```rust
trait WorkerAdapter {
    /// Spawn a new worker process
    fn spawn(&self, req: SpawnRequest) -> Result<WorkerId, SpawnError>;
    
    /// Check worker health (adapter may use HTTP, IPC, or other protocols)
    fn health_check(&self, worker_id: WorkerId) -> Result<WorkerHealth, HealthError>;
    
    /// Stop worker gracefully
    fn stop(&self, worker_id: WorkerId, grace_period: Duration) -> Result<(), StopError>;
    
    /// Force-kill worker
    fn kill(&self, worker_id: WorkerId) -> Result<(), KillError>;
    
    /// Get normalized worker state
    fn get_state(&self, worker_id: WorkerId) -> Result<WorkerState, StateError>;
    
    /// Adapter metadata (name, version, supported capabilities)
    fn metadata(&self) -> AdapterMetadata;
}
```

### [POOL-1002] Adapter Registry

Pool-managerd MUST maintain a registry of available adapters:
- Registry MUST be initialized at startup from configuration
- Registry MUST support dynamic adapter registration (optional, for plugins)
- Registry MUST validate adapter compatibility at registration time

### [POOL-1003] Adapter Selection

When orchestrator requests worker spawn with `worker_type` field, pool-managerd MUST:
1. Look up adapter by `worker_type` in registry
2. Delegate spawn operation to selected adapter
3. Return normalized worker state to orchestrator

If `worker_type` is omitted, pool-managerd SHOULD use default adapter (bespoke CUDA).

### [POOL-1004] Normalized Worker State

All adapters MUST normalize worker state to common format:
```rust
struct WorkerState {
    worker_id: String,
    status: WorkerStatus,  // starting, ready, busy, draining, failed
    capabilities: Vec<Capability>,  // text-gen, image-gen, etc.
    model_ref: String,
    vram_bytes: u64,
    uri: String,
    metadata: HashMap<String, String>,  // adapter-specific fields
}
```

### [POOL-1005] Capability Descriptors

Workers MUST advertise capabilities via adapters:
- `text-gen` — Text generation (LLMs)
- `image-gen` — Image generation (Stable Diffusion, DALL-E)
- `audio-gen` — Audio generation (speech synthesis, music)
- `embedding` — Text embeddings
- `multimodal` — Vision-language models

Orchestrator uses capabilities for job routing, not worker types.

---

## 3. Adapter Implementations

### 3.1 Bespoke CUDA Adapter (Default)

**Purpose**: Support existing `worker-orcd` binary (current implementation)

**Requirements**:
- [POOL-1010] Adapter MUST spawn `worker-orcd` with CLI args
- [POOL-1011] Adapter MUST handle ready callback at `/v2/internal/workers/ready`
- [POOL-1012] Adapter MUST translate worker health to normalized state
- [POOL-1013] Adapter MUST support VRAM accounting via NVML
- [POOL-1014] Adapter MUST advertise capability: `text-gen`

**Configuration**:
```yaml
adapters:
  - name: "bespoke-cuda"
    type: "BespokeCudaAdapter"
    binary_path: "/usr/local/bin/worker-orcd"
    default: true
    capabilities: ["text-gen"]
```

---

### 3.2 Llama.cpp Server Adapter

**Purpose**: Support external `llama.cpp` server as worker

**Requirements**:
- [POOL-1020] Adapter MUST spawn `llama-server` binary (from llama.cpp project)
- [POOL-1021] Adapter MUST translate llama.cpp `/health` to normalized state
- [POOL-1022] Adapter MUST translate llama.cpp `/completion` API to orchestrator execute contract
- [POOL-1023] Adapter MUST infer VRAM usage from `llama.cpp` memory reporting
- [POOL-1024] Adapter MUST advertise capability: `text-gen`

**Configuration**:
```yaml
adapters:
  - name: "llamacpp"
    type: "LlamaCppAdapter"
    binary_path: "/usr/local/bin/llama-server"
    capabilities: ["text-gen"]
```

**Protocol translation**:
- Orchestrator `/execute` → llama.cpp `/completion`
- llama.cpp `/health` → Normalized `WorkerHealth`
- llama.cpp SSE stream → Orchestrator SSE format

---

### 3.3 Apple Metal Adapter (ARM)

**Purpose**: Support Apple Silicon workers using Metal backend

**IMPORTANT**: Apple ARM workers use **UNIFIED MEMORY architecture**, NOT VRAM-only. This is a fundamental architectural difference from bespoke NVIDIA workers.

**Requirements**:
- [POOL-1030] Adapter MUST spawn `worker-armd` binary (Apple Metal worker with unified memory)
- [POOL-1031] Adapter MUST use Metal API for memory queries (not NVML)
- [POOL-1032] Adapter MUST support unified memory architecture (CPU and GPU share memory)
- [POOL-1033] Adapter MUST advertise capability: `text-gen`
- [POOL-1034] Adapter MUST report memory architecture as `unified` (not `vram-only`)

**Configuration**:
```yaml
adapters:
  - name: "apple-metal"
    type: "AppleMetalAdapter"
    binary_path: "/usr/local/bin/worker-aarmd"
    capabilities: ["text-gen"]
    memory_architecture: "unified"  # NOT vram-only
```

**Platform requirements**:
- Platform MUST be macOS or Linux on Apple Silicon
- Metal framework MUST be available

**Memory Architecture Note**: 
- Bespoke NVIDIA workers (`worker-orcd`): VRAM-ONLY policy (see `bin/.specs/01_M0_worker_orcd.md`)
- Apple ARM workers (`worker-aarmd`): UNIFIED MEMORY policy (see `bin/worker-aarmd/.specs/00_worker-aarmd.md` when created)

---

### 3.4 Image Generation Adapter

**Purpose**: Support Stable Diffusion / DALL-E workers for image generation

**Requirements**:
- [POOL-1040] Adapter MUST spawn image generation worker (e.g., `sd-worker`)
- [POOL-1041] Adapter MUST translate text prompt → image request
- [POOL-1042] Adapter MUST handle image output (base64, URL, binary)
- [POOL-1043] Adapter MUST advertise capability: `image-gen`

**Configuration**:
```yaml
adapters:
  - name: "stable-diffusion"
    type: "ImageGenAdapter"
    binary_path: "/usr/local/bin/sd-worker"
    capabilities: ["image-gen"]
    config:
      model_format: "safetensors"
      output_format: "base64"
```

**API contract extension**:
```json
{
  "model": "stable-diffusion-xl",
  "prompt": "a cat on a couch",
  "max_tokens": null,  // ignored for image-gen
  "width": 1024,
  "height": 1024,
  "steps": 50
}
```

---

## 4. Configuration Schema

### 4.1 Pool Manager Config Extension

```yaml
pool-managerd:
  bind: "0.0.0.0:9200"
  pool_id: "pool-1"
  
  # Worker adapter registry
  adapters:
    # Bespoke CUDA worker (default, backwards compatible)
    - name: "bespoke-cuda"
      type: "BespokeCudaAdapter"
      binary_path: "/usr/local/bin/worker-orcd"
      default: true
      capabilities: ["text-gen"]
    
    # Llama.cpp server
    - name: "llamacpp"
      type: "LlamaCppAdapter"
      binary_path: "/usr/local/bin/llama-server"
      capabilities: ["text-gen"]
      config:
        # llama.cpp-specific config
        context_size: 4096
    
    # Apple Metal worker
    - name: "apple-metal"
      type: "AppleMetalAdapter"
      binary_path: "/usr/local/bin/worker-orcd-arm"
      capabilities: ["text-gen"]
      platform: ["darwin"]  # macOS only
    
    # Stable Diffusion image generation
    - name: "stable-diffusion"
      type: "ImageGenAdapter"
      binary_path: "/usr/local/bin/sd-worker"
      capabilities: ["image-gen"]
```

### 4.2 Orchestrator Worker Selection

Orchestrator SHOULD select worker based on capability, not adapter type:

```rust
// Job requires "text-gen" capability
let job = Job { capability: Capability::TextGen, ... };

// Orchestrator queries pools for workers with "text-gen" capability
// Pool manager returns workers from ANY adapter that advertises "text-gen"
// (bespoke-cuda, llamacpp, apple-metal all match)
```

---

## 5. API Contract Extensions

### 5.1 Worker Start Request (Extended)

Orchestrator → Pool Manager:
```json
POST /v2/workers/start
{
  "model_ref": "hf:author/repo@rev::file=models/llama-7b.Q4_K_M.gguf",
  "gpu_id": 0,
  "worker_type": "llamacpp",  // NEW: optional, defaults to "bespoke-cuda"
  "capability": "text-gen"     // NEW: optional, for validation
}
```

### 5.2 Worker State Response (Extended)

Pool Manager → Orchestrator:
```json
GET /v2/state
{
  "pool_id": "pool-1",
  "gpus": [...],
  "workers": [
    {
      "id": "worker-abc",
      "worker_type": "llamacpp",  // NEW: adapter name
      "capabilities": ["text-gen"], // NEW: advertised capabilities
      "model_ref": "llama-7b",
      "gpu": 0,
      "vram_used": 16000000000,
      "uri": "http://localhost:8001",
      "status": "ready"
    }
  ]
}
```

---

## 6. Migration & Backwards Compatibility

### [POOL-1050] Backwards Compatibility

Pool-managerd MUST maintain backwards compatibility with existing deployments:
- If `adapters` config is omitted, pool-managerd MUST use default `BespokeCudaAdapter`
- If orchestrator omits `worker_type` in spawn request, pool-managerd MUST use default adapter
- Existing worker-orcd binaries MUST work without code changes

### [POOL-1051] Migration Path

Existing deployments can adopt adapters incrementally:
1. **Phase 1**: Deploy updated pool-managerd with default `BespokeCudaAdapter` (no config changes)
2. **Phase 2**: Add new adapter configs for alternative worker types
3. **Phase 3**: Orchestrator starts requesting specific `worker_type` per job

---

## 7. Implementation Roadmap

### 7.1 Milestone Dependencies

**Pre-requisite**: M2 (Orchestrator scheduling) MUST be complete

**Adapter milestones**:
- **M3.5** (Adapter foundation) — Implement `WorkerAdapter` trait, `BespokeCudaAdapter` (refactor existing logic)
- **M4.0** (External engines) — Implement `LlamaCppAdapter`, validation tests
- **M4.5** (Specialized hardware) — Implement `AppleMetalAdapter`, Metal VRAM queries
- **M5.0** (Multi-modality) — Implement `ImageGenAdapter`, capability-based routing

### 7.2 Implementation Order

1. Define `WorkerAdapter` trait in `bin/pool-managerd/src/adapters/mod.rs`
2. Refactor existing worker spawn logic into `BespokeCudaAdapter`
3. Implement adapter registry and config loading
4. Update `worker-lifecycle` to use adapters
5. Add tests: adapter registration, spawn delegation, state normalization
6. Implement `LlamaCppAdapter` (first external adapter)
7. Implement capability-based scheduling in orchestrator
8. Document adapter development guide for third-party vendors

---

## 8. Testing Strategy

### [POOL-1060] Adapter Testing

Each adapter implementation MUST have:
- **Unit tests**: Adapter spawn logic, state normalization, health translation
- **Integration tests**: End-to-end worker lifecycle via adapter
- **Contract tests**: Verify adapter normalizes worker state to expected schema
- **Interoperability tests**: Verify orchestrator can schedule across adapter types

### [POOL-1061] Capability-Based Routing Tests

Orchestrator MUST have tests verifying:
- Jobs with `text-gen` capability routed to any compatible worker (bespoke, llama.cpp, ARM)
- Jobs with `image-gen` capability routed only to image-gen workers
- Jobs with unknown capabilities rejected with clear error

---

## 9. Security Considerations

### [POOL-1070] Adapter Binary Validation

Pool-managerd MUST validate adapter binaries at startup:
- Binary path MUST exist and be executable
- Binary SHOULD be checksummed (optional, for integrity verification)
- Binary MUST NOT be writable by pool-managerd process (prevent tampering)

### [POOL-1071] Adapter Sandboxing

Workers spawned via adapters SHOULD be sandboxed:
- Use process isolation (already enforced by architecture)
- Use cgroups for resource limits (optional)
- Use network namespaces for isolation (optional, platform mode)

### [POOL-1072] Adapter Configuration Validation

Adapter configs MUST be validated:
- `binary_path` MUST be absolute path
- `capabilities` MUST be non-empty and from known set
- Platform restrictions (`platform: ["darwin"]`) MUST be enforced

---

## 10. Observability

### [POOL-1080] Adapter Metrics

Pool-managerd MUST emit adapter-specific metrics:
- `pool_mgr_adapter_spawns_total{adapter_name, outcome}` — Spawn attempts per adapter
- `pool_mgr_adapter_workers_total{adapter_name, status}` — Worker count per adapter
- `pool_mgr_adapter_health_checks_total{adapter_name, outcome}` — Health check results

### [POOL-1081] Adapter Logging

Adapter operations MUST be logged with:
- `adapter_name` — Adapter used for operation
- `worker_type` — Worker type being spawned
- `capabilities` — Advertised worker capabilities
- Spawn command and args (for debugging)

---

## 11. Future Extensions

### 11.1 Plugin System

**Future feature** (post-M5): Allow third-party adapter plugins:
- Adapters loaded as dynamic libraries (`.so`, `.dylib`, `.dll`)
- Adapter ABI versioning for compatibility
- Adapter signing/verification for security

### 11.2 Multi-Adapter Workers

**Future feature**: Single worker exposing multiple capabilities:
- Multimodal models (text + image)
- Adapter advertises multiple capabilities: `["text-gen", "image-gen"]`

### 11.3 Adapter Healthchecks

**Future feature**: Adapter-level health monitoring:
- Detect adapter failures (e.g., binary missing, protocol mismatch)
- Fallback to alternative adapter on failure
- Adapter health exposed in pool state

---

## 12. Traceability

**Parent spec**: `bin/.specs/00_llama-orch.md` (SYS-6.2.x pool-managerd)  
**Related specs**:
- `bin/pool-managerd/.specs/00_pool-managerd.md` (POOL-2xxx)
- `bin/.specs/00_llama-orch.md` (SYS-6.3.x worker contract)

**Implementation**:
- `bin/pool-managerd/src/adapters/` — Adapter trait and implementations
- `bin/pool-managerd/src/adapters/bespoke_cuda.rs` — Default adapter
- `bin/pool-managerd/src/adapters/llamacpp.rs` — Llama.cpp adapter
- `bin/pool-managerd/src/adapters/registry.rs` — Adapter registry

**Tests**:
- `bin/pool-managerd/tests/adapters/` — Adapter tests
- `bin/pool-managerd/bdd/features/worker_adapters.feature` — BDD scenarios

---

## 13. Decision Log

### 13.1 Why Adapters in Pool Manager, Not Orchestrator?

**Decision**: Worker adapters live in pool-managerd, not orchestratord

**Rationale**:
- Pool manager owns worker spawning (FFI, process management)
- Orchestrator remains agnostic to worker implementation details
- Supports heterogeneous pools (different adapters per pool)
- Preserves smart/dumb boundary (orchestrator = smart, pool = dumb executor with many levers)

### 13.2 Why Capability-Based Routing?

**Decision**: Orchestrator routes by capability, not worker type

**Rationale**:
- Future-proof: New adapters don't require orchestrator changes
- Flexible: Multiple adapters can satisfy same capability
- Semantically correct: Jobs describe what they need, not how to run

### 13.3 Why Not Nested Adapters?

**Decision**: Flat adapter registry, no adapter composition

**Rationale**:
- YAGNI (You Aren't Gonna Need It) — No current use case for nested adapters
- Simplicity: Flat registry is easier to reason about and debug
- Can add composition later if needed without breaking changes

---

**End of Specification**
