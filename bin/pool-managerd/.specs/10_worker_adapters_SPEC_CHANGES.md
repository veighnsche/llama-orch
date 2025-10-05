# Spec Changes Required for Worker Adapter Support

**Date**: 2025-10-05  
**Status**: Planning Document  
**Related**: `10_worker_adapters.md`

---

## Overview

This document outlines the changes needed to existing specs to prepare for worker adapter support. The adapter pattern is a **future feature** (post-M2) but requires spec alignment now to avoid breaking changes later.

---

## 1. Changes to `bin/.specs/00_llama-orch.md` (System Spec)

### 1.1 Add Worker Type Concept

**Location**: Section 0.1 (Glossary)

**Add new term**:
```markdown
- **Worker Type**: Category of worker implementation (bespoke-cuda, llamacpp, apple-metal, image-gen). Pool manager uses worker type to select appropriate adapter for spawning.
- **Worker Capability**: Functional capability advertised by worker (text-gen, image-gen, audio-gen, embedding). Orchestrator routes jobs based on capability, not worker type.
- **Worker Adapter**: Component in pool manager that abstracts worker lifecycle for specific worker type. Adapters normalize worker state and translate protocols.
```

### 1.2 Update Worker Contract (SYS-6.3.x)

**Location**: Section 6.3 (Worker-Orcd)

**Current limitation**:
```markdown
### 6.3 Worker-Orcd (Executor) [M0] (SYS-6.3.x)

**Binary**: `bin/worker-orcd/`
```

**Proposed change**:
```markdown
### 6.3 Worker Contract (Executor) [M0] (SYS-6.3.x)

**Default implementation**: `bin/worker-orcd/` (bespoke CUDA worker)  
**Extensibility**: Pool manager MAY support alternative worker implementations via adapter pattern (see POOL-1xxx spec)

**Worker contract requirements** (all workers MUST satisfy):
- [SYS-6.3.1] Worker isolation (separate processes)
- [SYS-6.3.2] Ready callback contract
- [SYS-6.3.3] HTTP API endpoints (/execute, /cancel, /health)
- [SYS-6.3.4] VRAM reporting (or equivalent resource reporting)
- [SYS-6.3.5] Capability advertisement (text-gen, image-gen, etc.)

**Bespoke CUDA worker** (`worker-orcd`):
- Implementation details in `bin/worker-orcd/.specs/00_worker-orcd.md`
- VRAM-only policy, CUDA kernels, GGUF format support
```

**Rationale**: Future workers may not use CUDA (e.g., Apple Metal), but must satisfy common contract.

### 1.3 Update Pool Manager Responsibilities (SYS-6.2.x)

**Location**: Section 6.2.2 (Pool Manager Execution)

**Add requirement**:
```markdown
#### [SYS-6.2.6] Worker Adapter Support (Future)

Pool-managerd MAY use worker adapters to support multiple worker implementations:
- Default adapter MUST support bespoke CUDA worker (backwards compatibility)
- Additional adapters MAY be configured for llama.cpp, Apple Metal, image generation
- Adapter selection MUST be based on `worker_type` field in spawn request
- All workers MUST be normalized to common state format via adapter interface

**Spec**: `bin/pool-managerd/.specs/10_worker_adapters.md` (POOL-1xxx)
```

### 1.4 Update API Contract (SYS-5.3.x)

**Location**: Section 5.3 (Pool Manager ↔ Worker)

**Extend worker start request**:
```markdown
#### [SYS-5.3.2] Worker Start Request

POST /v2/workers/start
{
  "model_ref": "hf:author/repo@rev::file=models/llama-7b.Q4_K_M.gguf",
  "gpu_id": 0,
  "worker_type": "bespoke-cuda",  // OPTIONAL: defaults to "bespoke-cuda" (M0-M2), adapter name (M3+)
  "capability": "text-gen"         // OPTIONAL: for validation (M3+)
}

**Requirements**:
- `model_ref` MUST be normalized (see SYS-2.1.2)
- `gpu_id` MUST refer to valid GPU on pool node
- `worker_type` MAY be omitted; pool manager MUST use default adapter
- `capability` MAY be used for preflight validation (M3+)
```

**Extend worker state response**:
```markdown
#### [SYS-5.3.3] Worker State Response

GET /v2/state returns:
{
  "workers": [
    {
      "id": "worker-abc",
      "worker_type": "bespoke-cuda",    // NEW (M3+): adapter name
      "capabilities": ["text-gen"],     // NEW (M3+): advertised capabilities
      "model_ref": "...",
      "status": "ready",
      ...
    }
  ]
}
```

---

## 2. Changes to `bin/pool-managerd/.specs/00_pool-managerd.md`

### 2.1 Add Adapter Section Reference

**Location**: Section 0 (Scope)

**Add forward reference**:
```markdown
### Related Specs

- **Worker Adapters** (`10_worker_adapters.md`, POOL-1xxx) — Pluggable worker backend support (future feature, post-M2)
```

### 2.2 Update Worker Lifecycle (Section 5)

**Location**: Section 5 (Worker Lifecycle), [POOL-2040]

**Current**:
```markdown
### [POOL-2040] Start Worker Command

5. **Spawn process**: Start `worker-orcd` process with args:
   ```bash
   worker-orcd \
     --worker-id worker-{uuid} \
     --model /path/to/model.gguf \
     --gpu-device {gpu_id} \
     --port {port} \
     --callback-url http://localhost:9200/v2/internal/workers/ready
   ```
```

**Proposed**:
```markdown
### [POOL-2040] Start Worker Command

5. **Spawn process**: Delegate to worker adapter (default: BespokeCudaAdapter)
   
   **M0-M2 implementation** (hardcoded):
   ```bash
   worker-orcd \
     --worker-id worker-{uuid} \
     --model /path/to/model.gguf \
     --gpu-device {gpu_id} \
     --port {port} \
     --callback-url http://localhost:9200/v2/internal/workers/ready
   ```
   
   **M3+ implementation** (adapter-based):
   - Look up adapter by `worker_type` (from spawn request)
   - Delegate spawn to adapter.spawn(request)
   - Adapter handles binary path, CLI args, protocol translation
   
   **See**: `10_worker_adapters.md` (POOL-1xxx) for adapter spec
```

### 2.3 Add Configuration Section for Adapters

**Location**: Section 9 (Configuration)

**Add subsection**:
```markdown
### [POOL-2082] Worker Adapter Config (M3+)

Pool-managerd MAY accept adapter registry configuration:

```yaml
adapters:
  - name: "bespoke-cuda"
    type: "BespokeCudaAdapter"
    binary_path: "/usr/local/bin/worker-orcd"
    default: true
    capabilities: ["text-gen"]
  
  - name: "llamacpp"
    type: "LlamaCppAdapter"
    binary_path: "/usr/local/bin/llama-server"
    capabilities: ["text-gen"]
```

**Requirements** (M3+):
- If `adapters` config is omitted, pool-managerd MUST use default `BespokeCudaAdapter`
- At least one adapter MUST be marked as `default: true`
- Adapter `binary_path` MUST be absolute path to executable
- Adapter `capabilities` MUST be non-empty list from known set

**See**: `10_worker_adapters.md` for full adapter config schema
```

---

## 3. Changes to `bin/worker-orcd/.specs/00_worker-orcd.md`

### 3.1 Clarify Scope

**Location**: Section 0 (Purpose)

**Add note**:
```markdown
### Scope Note

This spec describes the **bespoke CUDA worker** implementation (`worker-orcd`). This is the default worker for M0-M2.

Future worker types (llama.cpp, Apple Metal, image-gen) will have separate implementations but MUST satisfy the common worker contract defined in the parent spec (SYS-6.3.x).

**See**: `bin/pool-managerd/.specs/10_worker_adapters.md` for multi-worker support via adapters.
```

---

## 4. Model Reference Schema Extension

### 4.1 Support Non-LLM Models

**Location**: `bin/.specs/00_llama-orch.md`, Section 2.1 (Model Reference Format)

**Current limitation**:
```markdown
- `hf:{org}/{repo}@{rev}::file={path}` — Hugging Face models (GGUF format assumed)
```

**Proposed extension** (M4+):
```markdown
#### [SYS-2.1.4] Model Reference Extension for Non-LLM Models

Model references MAY include format hints for non-LLM models:

- `hf:org/repo@rev::file=model.gguf` — Text-gen model (GGUF)
- `hf:org/repo@rev::file=model.safetensors` — Image-gen model (Stable Diffusion)
- `hf:org/repo@rev::file=model.pt` — PyTorch model (generic)

**Requirements**:
- File extension SHOULD indicate model format for adapter selection
- Worker adapters MUST validate model format compatibility at preflight
- Orchestrator catalog SHOULD store model format metadata
```

---

## 5. Orchestrator Scheduling Extension

### 5.1 Capability-Based Routing

**Location**: `bin/.specs/00_llama-orch.md`, Section 6.1.5 (Programmable Scheduler)

**Add requirement**:
```markdown
#### [SYS-6.1.7] Capability-Based Scheduling (M4+)

Orchestrator scheduler SHOULD route jobs based on worker capabilities, not worker types:

**Requirements**:
- Job requests SHOULD specify required capability (e.g., `capability: "text-gen"`)
- Scheduler MUST only consider workers advertising required capability
- Scheduler MUST NOT hardcode worker type preferences
- Rhai scheduler scripts MAY access worker capabilities via `worker.capabilities`

**Example**:
```rhai
// Rhai scheduler: route text-gen jobs to any worker with "text-gen" capability
let text_gen_workers = workers.filter(|w| w.capabilities.contains("text-gen"));
let selected = least_loaded(text_gen_workers);
```

**Rationale**: Allows heterogeneous worker pools (bespoke, llama.cpp, ARM) to serve same job types.
```

---

## 6. OpenAPI Contract Changes

### 6.1 Worker Start Request

**Location**: `contracts/openapi/pool-manager.yaml`

**Add optional fields**:
```yaml
paths:
  /v2/workers/start:
    post:
      requestBody:
        content:
          application/json:
            schema:
              type: object
              required: [model_ref, gpu_id]
              properties:
                model_ref:
                  type: string
                  example: "hf:author/repo@rev::file=models/llama-7b.Q4_K_M.gguf"
                gpu_id:
                  type: integer
                  example: 0
                worker_type:  # NEW (optional)
                  type: string
                  description: "Worker adapter to use. Defaults to 'bespoke-cuda' if omitted."
                  example: "llamacpp"
                capability:  # NEW (optional)
                  type: string
                  description: "Required capability for validation (text-gen, image-gen, etc.)"
                  example: "text-gen"
```

### 6.2 Worker State Response

**Location**: `contracts/openapi/pool-manager.yaml`

**Extend worker object**:
```yaml
components:
  schemas:
    Worker:
      type: object
      required: [id, model_ref, status, uri]
      properties:
        id:
          type: string
        model_ref:
          type: string
        status:
          type: string
          enum: [starting, ready, busy, draining, failed]
        uri:
          type: string
        worker_type:  # NEW
          type: string
          description: "Adapter name used to spawn this worker"
          example: "bespoke-cuda"
        capabilities:  # NEW
          type: array
          items:
            type: string
          description: "Capabilities advertised by this worker"
          example: ["text-gen"]
```

---

## 7. Metrics Extensions

### 7.1 Pool Manager Adapter Metrics

**Location**: `bin/.specs/71_metrics_contract.md`

**Add metrics** (M3+):
```markdown
### Pool Manager Adapter Metrics

- `pool_mgr_adapter_spawns_total{adapter_name, outcome}` — Worker spawn attempts per adapter
  - Labels: `adapter_name` (bespoke-cuda, llamacpp, apple-metal), `outcome` (success, preflight_fail, timeout, crash)
  
- `pool_mgr_adapter_workers_total{adapter_name, status}` — Active workers per adapter
  - Labels: `adapter_name`, `status` (starting, ready, busy, failed)
  
- `pool_mgr_adapter_health_checks_total{adapter_name, outcome}` — Health check results per adapter
  - Labels: `adapter_name`, `outcome` (healthy, unhealthy, timeout)
```

---

## 8. Implementation Phases

### Phase 1: Spec Alignment (Now - Pre-M3)

**Goal**: Update specs to prepare for adapters without breaking M0-M2

**Actions**:
1. ✅ Write `10_worker_adapters.md` spec (this document)
2. ⬜ Add glossary terms to `00_llama-orch.md`
3. ⬜ Update worker contract section (SYS-6.3.x) to be implementation-agnostic
4. ⬜ Add optional `worker_type` field to API contracts (backwards compatible)
5. ⬜ Document forward references in pool-managerd spec

**Impact**: No code changes, spec preparation only

### Phase 2: Adapter Foundation (M3.5)

**Goal**: Refactor existing code into adapter pattern

**Actions**:
1. ⬜ Define `WorkerAdapter` trait
2. ⬜ Implement `BespokeCudaAdapter` (refactor existing spawn logic)
3. ⬜ Implement adapter registry (load from config)
4. ⬜ Update `worker-lifecycle` to use adapters
5. ⬜ Add tests: adapter registration, default adapter, backwards compatibility

**Impact**: Internal refactor, no user-facing changes

### Phase 3: External Adapters (M4.0+)

**Goal**: Add support for llama.cpp and other worker types

**Actions**:
1. ⬜ Implement `LlamaCppAdapter`
2. ⬜ Protocol translation tests (llama.cpp ↔ orchestrator API)
3. ⬜ Integration tests: mixed pool (bespoke + llama.cpp workers)
4. ⬜ Document adapter development guide

**Impact**: New worker types supported, existing deployments unaffected

### Phase 4: Capability Routing (M4.5+)

**Goal**: Orchestrator routes by capability, not worker type

**Actions**:
1. ⬜ Extend Rhai scheduler with capability awareness
2. ⬜ Update admission to validate capability requirements
3. ⬜ Add metrics: jobs_routed_by_capability{capability, worker_type}

**Impact**: Heterogeneous pools can serve same job types

---

## 9. Backwards Compatibility Guarantees

### 9.1 M0-M2 Deployments

**Guarantee**: Existing deployments continue to work without config changes

**How**:
- If `adapters` config is omitted, pool-managerd uses default `BespokeCudaAdapter`
- If `worker_type` is omitted in spawn request, pool-managerd uses default adapter
- Existing `worker-orcd` binary works without modification

### 9.2 API Compatibility

**Guarantee**: New API fields are optional

**How**:
- `worker_type` field in spawn request is optional (defaults to "bespoke-cuda")
- `capabilities` field in state response is optional (defaults to `["text-gen"]`)
- Old clients continue to work with new pool-managerd

### 9.3 Configuration Migration

**Guarantee**: No breaking config changes

**How**:
- `adapters` section is optional (uses defaults if omitted)
- Existing pool-managerd configs continue to work
- New configs can opt-in to adapters incrementally

---

## 10. Open Questions

### 10.1 Adapter Plugin System

**Question**: Should we support dynamically loaded adapter plugins (`.so` libraries)?

**Options**:
- **Static linking** (M3-M4): Adapters compiled into pool-managerd binary
- **Dynamic plugins** (M5+): Adapters loaded as shared libraries at runtime

**Recommendation**: Start with static linking for simplicity. Add plugin system in M5+ if needed.

### 10.2 Cross-Adapter Scheduling

**Question**: Can orchestrator schedule same job to different adapter types?

**Example**: Job says "text-gen" → orchestrator can pick bespoke OR llama.cpp worker

**Recommendation**: YES, this is the goal of capability-based routing. Scheduler sees only capabilities, not adapter types.

### 10.3 Adapter Versioning

**Question**: How do we handle adapter version compatibility?

**Options**:
- Adapter metadata includes version (e.g., `BespokeCudaAdapter v1.0`)
- Pool-managerd validates adapter version at registration
- Breaking adapter changes require major version bump

**Recommendation**: Add `version` field to `AdapterMetadata` in M3.5.

---

## 11. Summary

**Spec changes needed**:
1. ✅ New spec: `10_worker_adapters.md` (POOL-1xxx)
2. ⬜ Update `00_llama-orch.md` glossary (worker type, capability, adapter)
3. ⬜ Update `00_llama-orch.md` SYS-6.3.x (worker contract vs. bespoke implementation)
4. ⬜ Update `00_pool-managerd.md` POOL-2040 (adapter-based spawn)
5. ⬜ Update OpenAPI contracts (optional worker_type, capabilities fields)
6. ⬜ Update metrics contract (adapter-specific metrics)

**Implementation phases**:
- **Now**: Spec alignment (no code changes)
- **M3.5**: Adapter foundation (internal refactor)
- **M4.0**: External adapters (llama.cpp)
- **M4.5**: Capability routing

**Backwards compatibility**:
- Existing deployments continue to work
- New fields are optional
- Default adapter preserves M0-M2 behavior

---

**End of Document**
