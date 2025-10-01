# Terminology Formalization Proposal: Three-Binary Architecture

**Date**: 2025-10-01  
**Author**: Engineering Team  
**Status**: PROPOSAL — Awaiting Review  
**Purpose**: Establish canonical terminology to eliminate confusion in naming, documentation, and developer ergonomics

---

## Executive Summary

This document formalizes **all major terminology** for the llama-orch three-binary architecture to prevent naming confusion, overlapping responsibilities, and developer cognitive load.

**Key Proposal**: We recommend **renaming one crate and clarifying all terminology** before the worker-orcd implementation begins.

---

## PART A: Binary-Level Naming (The Three Daemons)

### Current State

```
orchestratord       → Orchestrator (control plane, no GPU)
pool-managerd       → Pool manager (per-host, GPU lifecycle)
worker-orcd         → Worker (per-GPU, inference execution)
```

### Analysis: Names Are Good ✅

**Recommendation**: **KEEP AS-IS**

**Rationale**:

- Clear `-d` suffix convention (daemon binaries)
- Distinct responsibilities with no overlap
- Memorable brand story alignment (see `BRAND_STORY_THREE_ORCS.md`)
- Consistent with Unix daemon naming conventions

**No changes needed.**

---

## PART B: Architectural Layers (Vertical Granularity)

### Problem Statement

We use overlapping terms to describe different **levels of granularity**:

- **Physical**: Hardware (GPU cards, machines)
- **Logical**: Software abstractions (pools, replicas, workers, shards)
- **Network**: Multi-node deployments (nodes, clusters)

This causes confusion in naming libraries, crates, and documentation.

### Proposed Canonical Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│ CLUSTER (multi-node deployment)                             │
│   ├─ orchestratord (1 per cluster, or N for HA)            │
│   └─ NODE (physical machine)                               │
│       ├─ pool-managerd (1 per node)                        │
│       └─ GPU DEVICE (physical GPU card, e.g., GPU 0)       │
│           └─ WORKER (worker-orcd process, 1 per GPU)       │
│               └─ POOL (logical unit, 1 model loaded)       │
│                   └─ REPLICA (execution slot within pool)  │
│                       └─ SHARD (tensor-parallel piece)     │
└─────────────────────────────────────────────────────────────┘
```

### Definitions (NORMATIVE)

| Term | Granularity | Definition | Example |
|------|-------------|------------|---------|
| **Cluster** | Deployment | All llama-orch components working together | Production deployment |
| **Node** | Physical | One physical machine (bare metal or VM) | `gpu-node-1.local` |
| **GPU Device** | Physical | One NVIDIA GPU card | `CUDA_VISIBLE_DEVICES=0` |
| **Worker** | Process | One `worker-orcd` process bound to 1 GPU | `worker-orcd --gpu=0` |
| **Pool** | Logical | One model loaded into VRAM on 1 GPU | `pool-0` (Llama-3.1-8B) |
| **Replica** | Execution | One concurrent inference slot within a pool | 4 replicas = 4 jobs in parallel |
| **Shard** | Tensor-Parallel | One piece of a TP-sharded model | Llama-70B shard 0/4 |

### Key Distinctions

**Node vs Worker**:

- **Node** = Physical machine (runs `pool-managerd`)
- **Worker** = Process on that node (runs `worker-orcd`, one per GPU)

**Pool vs Replica**:

- **Pool** = Model loaded into VRAM (e.g., "Llama-3.1-8B on GPU 0")
- **Replica** = Execution slot (e.g., "4 concurrent jobs in that pool")

**Worker vs Shard**:

- **Worker** = Entire process managing 1 GPU
- **Shard** = Piece of a TP model distributed across multiple workers

---

## PART C: Registry/Tracking Crates (THE CORE ISSUE)

### Current Naming Confusion ❌

We have **THREE different "registries"** with overlapping names:

1. **`service-registry`** (tracks GPU **nodes** in orchestratord)
2. **`node-registration`** (client for GPU **nodes** to register with orchestratord)
3. **`pool-managerd::core::Registry`** (tracks **pools** within one node)

**Problem**: Two of these use "registry" but track different things (nodes vs pools).

### Root Cause Analysis

**Confusion arises because**:

- `service-registry` tracks **nodes** but is named generically
- `node-registration` is a **client** but sounds like a server
- `pool-managerd::Registry` tracks **pools** but lives inside pool-managerd

---

### Proposed Renaming (CRITICAL CHANGE)

#### Option A: Emphasize Node vs Pool Distinction (RECOMMENDED)

| Current Name | New Name | Tracks | Used By |
|--------------|----------|--------|---------|
| `service-registry` | **`node-registry`** | GPU nodes | orchestratord |
| `node-registration` | **`node-registration-client`** | N/A (client) | pool-managerd |
| `pool-managerd::Registry` | **`pool-registry`** (in-process) | Pools on this node | pool-managerd |

**Rationale**:

- **`node-registry`** = explicitly tracks nodes
- **`node-registration-client`** = explicitly a client
- **`pool-registry`** = explicitly tracks pools

**File paths**:

```
bin/pool-managerd-crates/
  ├─ node-registration-client/     # Client for registering nodes
  └─ node-registry/                  # Server-side node tracking

bin/pool-managerd/src/core/
  └─ pool_registry.rs                # Pool-level tracking
```

---

#### Option B: Emphasize Client/Server Distinction

| Current Name | New Name | Tracks | Used By |
|--------------|----------|--------|---------|
| `service-registry` | **`control-plane-registry`** | GPU nodes | orchestratord |
| `node-registration` | **`control-plane-client`** | N/A (client) | pool-managerd |
| `pool-managerd::Registry` | **`pool-tracker`** (in-process) | Pools on this node | pool-managerd |

**Rationale**:

- Emphasizes control plane boundary
- `client` suffix makes directionality explicit

---

#### Option C: Minimal Change

| Current Name | New Name | Tracks | Used By |
|--------------|----------|--------|---------|
| `service-registry` | **`node-registry`** | GPU nodes | orchestratord |
| `node-registration` | `node-registration` (KEEP) | N/A (client) | pool-managerd |
| `pool-managerd::Registry` | `pool-managerd::Registry` (KEEP) | Pools on this node | pool-managerd |

**Rationale**:

- Least disruptive
- Only fixes the most ambiguous name

---

### **RECOMMENDATION**: **Option A** (Node/Pool Distinction)

**Justification**:

1. Most explicit about what each crate tracks
2. Aligns with architectural hierarchy (nodes contain pools)
3. `-client` suffix is industry standard (e.g., `etcd-client`, `k8s-client`)
4. Easier to explain to new developers

---

## PART D: Shared Types Library

### Current State

**`pool-registry-types`** — Shared types for node/pool communication

### Problem

The name `pool-registry-types` implies it's only for pools, but it actually contains:

- **Node types** (`NodeInfo`, `NodeCapabilities`)
- **Pool types** (`PoolSnapshot`)
- **GPU types** (`GpuInfo`)
- **Health types** (`HealthStatus`)

### Proposed Rename

**Option 1: `cluster-types`** (RECOMMENDED)

- Rationale: Types used across the entire cluster (nodes + pools)
- Clear scope: "If it crosses network boundaries, it goes here"

**Option 2: `node-pool-types`**

- Rationale: Explicitly mentions both node and pool
- Con: Verbose

**Option 3: `llama-orch-types`**

- Rationale: Generic shared types
- Con: Too broad, loses specificity

**RECOMMENDATION**: **`cluster-types`**

---

## PART E: Worker-orcd Component Naming

### Current State (from ARCHITECTURE_CHANGE_PLAN.md)

```
worker-orcd/
  ├─ src/
  │   ├─ lifecycle.rs
  │   ├─ residency.rs
  │   ├─ telemetry.rs
  │   ├─ scheduler.rs
  │   ├─ server.rs
  │   ├─ cuda_ffi.rs
  │   └─ engine.rs
  └─ cuda/
      ├─ gemm.cu
      ├─ rope.cu
      └─ attention.cu
```

### Proposed Terminology Alignment

| Module | Canonical Term | Purpose |
|--------|----------------|---------|
| `residency.rs` | **VRAM Residency Enforcement** | Ensures no RAM inference |
| `engine.rs` | **Inference Engine** | Forward pass execution |
| `server.rs` | **RPC Server** | Plan/Commit/Ready/Execute endpoints |
| `cuda_ffi.rs` | **FFI Boundary** | Rust ↔ CUDA interface |

**Key Term**: **ModelShardHandle** (sealed VRAM shard abstraction)

---

## PART F: Deployment Mode Terminology (NOT Cloud vs Home)

### Current Terms (DEPRECATED)

- ❌ **Cloud Profile** — Misleading (implies datacenter/commercial)
- ❌ **Home Profile** — Misleading (home labs can have multiple nodes)

### Proposed Canonical Terms

| Term | Definition | Architecture |
|------|------------|--------------|
| **Single-Node Mode** | 1 physical machine, 1+ GPUs | orchestratord + pool-managerd |
| **Multi-Node Mode** | 2+ physical machines | orchestratord + multiple pool-managerd instances |
| **Node** | One physical machine | `gpu-node-1`, `gpu-node-2` |
| **Control Plane** | orchestratord + node registries | Placement, admission, streaming |
| **Data Plane** | pool-managerd + workers | Model execution, VRAM management |

**Key Distinctions**:

- **Single-Node Mode**: 1 machine with GPUs (e.g., workstation with 2x RTX 4090)
- **Multi-Node Mode**: 2+ machines (e.g., home lab with 3 GPU servers)

**Important**: A **home lab with 3 GPU machines** is **Multi-Node Mode**, not "Cloud Profile"! The distinction is **architectural** (node count), not **deployment context** (cloud vs home).

---

## PART G: Tensor-Parallel Terminology

### Current Terms (from ARCHITECTURE_CHANGE_PLAN.md)

- **Shard** — Piece of a TP model
- **NCCL Group** — Communicator for cross-GPU ops

### Proposed Canonical Terms

| Term | Definition | Example |
|------|------------|---------|
| **Tensor-Parallel (TP)** | Sharding model across GPUs | Llama-70B on 4 GPUs |
| **Shard** | One piece of TP model | Layers 0-19 on GPU 0 |
| **NCCL Group** | Cross-GPU communicator | Workers coordinate via NCCL |
| **Shard Layout** | Plan for distributing shards | [GPU 0: layers 0-19, GPU 1: layers 20-39] |

**Avoid**:

- ❌ "Partition" (confusing with data partitioning)
- ❌ "Slice" (ambiguous)
- ✅ "Shard" (clear, industry standard)

---

## PART H: Capability Matching Terminology

### Current Terms (from ARCHITECTURE_CHANGE_PLAN.md)

- **MCD** — Model Capability Descriptor
- **ECP** — Engine Capability Profile

### Proposed Canonical Terms

| Term | Definition | Where |
|------|------------|-------|
| **MCD** (Model Capability Descriptor) | Metadata in model artifact | Embedded in GGUF or sidecar JSON |
| **ECP** (Engine Capability Profile) | Worker's advertised capabilities | worker-orcd config or introspection |
| **Capability Matching** | MCD ⊆ ECP check | Placement decision in orchestratord |

**Key Phrase**: "Model requires X, worker supports Y. Compatible if X ⊆ Y."

---

## PART I: Authentication/Security Terminology

### Current Terms (from specs)

- **Minimal Auth Hooks** (AUTH-1xxx)
- **Bearer Token**
- **Loopback Bypass**

### Proposed Canonical Terms

| Term | Definition | Example |
|------|------------|---------|
| **Bearer Token** | API authentication token | `Authorization: Bearer secret123` |
| **Loopback Bypass** | Skip auth for localhost | `--bind=127.0.0.1` allows no token |
| **Identity Fingerprint** | First 6 hex chars of token SHA-256 | `fp6=a1b2c3` in logs |
| **Control Plane Auth** | orchestratord → pool-managerd | Bearer token over HTTP |
| **Data Plane Auth** | pool-managerd → worker-orcd | Bearer token or mTLS |

**Key Distinction**:

- **Control Plane**: North-south (client → orchestratord)
- **Data Plane**: East-west (orchestratord → pool-managerd → worker-orcd)

---

## PART J: Directory Structure Alignment

### Proposed Standard Paths

```
bin/
  ├─ orchestratord/              # Control plane daemon
  ├─ pool-managerd/              # Data plane daemon (per-node)
  │   └─ src/core/
  │       └─ pool_registry.rs    # Pool-level tracking
  └─ worker-orcd/                # Worker daemon (per-GPU)

bin/orchestratord-crates/
  ├─ orchestrator-core/          # Admission, queueing, placement
  └─ node-registry/              # ← RENAMED from service-registry
      └─ src/
          ├─ lib.rs              # Node tracking state
          ├─ api.rs              # Registration payloads
          └─ heartbeat.rs        # Stale node detection

bin/pool-managerd-crates/
  ├─ catalog-core/               # Model catalog
  ├─ model-provisioner/          # Model staging
  └─ node-registration-client/   # ← RENAMED from node-registration
      └─ src/
          ├─ lib.rs              # Registration client
          └─ client.rs           # HTTP client to orchestratord

libs/
  ├─ cluster-types/              # ← RENAMED from pool-registry-types
  │   └─ src/
  │       ├─ node.rs             # NodeInfo, NodeCapabilities
  │       ├─ pool.rs             # PoolSnapshot
  │       ├─ gpu.rs              # GpuInfo
  │       └─ health.rs           # HealthStatus
  └─ auth-min/                   # Authentication primitives
```

---

## PART K: Documentation Conventions

### Proposed Standard Language

**When referring to binaries**:

- ✅ "orchestratord (the orchestrator daemon)"
- ✅ "pool-managerd (the pool manager daemon)"
- ✅ "worker-orcd (the worker daemon)"
- ❌ "the orchestrator" (ambiguous: daemon or component?)

**When referring to layers**:

- ✅ "control plane" (lowercase, generic)
- ✅ "data plane" (lowercase, generic)
- ❌ "Control Plane" (not a proper noun)

**When referring to abstractions**:

- ✅ "a node" (one physical machine)
- ✅ "a worker" (one worker-orcd process)
- ✅ "a pool" (one model loaded on one GPU)
- ✅ "a replica" (one execution slot in a pool)

**When referring to VRAM**:

- ✅ "VRAM" (all caps, GPU memory)
- ✅ "RAM" (all caps, host memory)
- ❌ "vram" (lowercase is informal)

---

## PART L: Email-Style Summary for Stakeholders

---

**Subject**: Terminology Proposal for Three-Binary Architecture — Developer Ergonomics

**To**: Engineering Team, Product, Documentation  
**From**: Architecture Review  
**Date**: 2025-10-01

Hi team,

We've completed a comprehensive terminology audit for the three-binary architecture (orchestratord / pool-managerd / worker-orcd). This proposal aims to **eliminate naming confusion** before we begin the worker-orcd implementation.

### The Core Issue

We currently have **overlapping "registry" names** that track different things:

1. **`service-registry`** — Tracks GPU **nodes** (used by orchestratord)
2. **`node-registration`** — Client for **nodes** to register with orchestratord
3. **`pool-managerd::Registry`** — Tracks **pools** within one node

This causes confusion: "Which registry tracks what?"

### Our Recommendation

**Rename two crates to emphasize node vs. pool distinction**:

| Old Name | New Name | What It Tracks |
|----------|----------|----------------|
| `service-registry` | **`node-registry`** | GPU nodes (server-side) |
| `node-registration` | **`node-registration-client`** | Registration client |
| `pool-registry-types` | **`cluster-types`** | Shared types (nodes + pools) |

**Leave as-is**:

- `pool-managerd::Registry` → Already clear (tracks pools)
- Daemon names (`orchestratord`, `pool-managerd`, `worker-orcd`) → Already good

### Why This Matters

**Before** (confusing):

- Developer: "Wait, does `service-registry` track services or nodes?"
- Developer: "Why is `node-registration` a client if it sounds like a server?"
- Developer: "Why are node types in a library called `pool-registry-types`?"

**After** (clear):

- `node-registry` → Obviously tracks nodes
- `node-registration-client` → Obviously a client
- `cluster-types` → Obviously shared cluster-wide types

### Additional Clarity

We've also formalized:

1. **Architectural Hierarchy** (from largest to smallest):
   - Cluster → Node → GPU → Worker → Pool → Replica → Shard

2. **Multi-Node Terms**:
   - Control Plane = orchestratord + node tracking
   - Data Plane = pool-managerd + workers

3. **Worker Terms**:
   - ModelShardHandle = Sealed VRAM shard abstraction
   - MCD/ECP = Capability matching for model compatibility

4. **Docs Conventions**:
   - "orchestratord" (lowercase, with `-d`)
   - "control plane" (lowercase, generic)
   - "VRAM" (all caps)

### Impact

**Code changes required**:

- Rename `service-registry/` → `node-registry/`
- Rename `node-registration/` → `node-registration-client/`
- Rename `pool-registry-types/` → `cluster-types/`
- Update imports in orchestratord, pool-managerd, worker-orcd
- Update Cargo.toml workspace members

**Estimated effort**: 2-3 hours (mostly find-replace, plus compile checks)

**Benefit**: Eliminates major naming confusion **before** we start worker-orcd implementation.

### Decision Requested

Please review and approve (or suggest alternatives) by EOD. We want to make these changes **now** while we're pre-v1.0 and destructive changes are allowed.

**Options**:

1. ✅ Approve Option A (Node/Pool Distinction) — **RECOMMENDED**
2. 🔄 Request Option B (Client/Server Distinction)
3. 🔄 Request Option C (Minimal Change)
4. ❌ Reject (keep current names)

Thanks,  
Architecture Team

---

## PART M: Implementation Checklist

### Phase 1: Rename Crates (2-3 hours)

- [ ] Rename `bin/pool-managerd-crates/both_registers_question_mark/service-registry/` → `node-registry/`
- [ ] Rename `bin/pool-managerd-crates/both_registers_question_mark/node-registration/` → `node-registration-client/`
- [ ] Rename `libs/pool-registry-types/` → `libs/cluster-types/`
- [ ] Update `Cargo.toml` workspace members
- [ ] Update all `Cargo.toml` dependencies (`service-registry` → `node-registry`, etc.)
- [ ] Update all `use` imports in Rust files
- [ ] Update all README.md files in renamed crates
- [ ] Run `cargo check --workspace` to verify compilation

### Phase 2: Update Documentation (1-2 hours)

- [ ] Update root `README.md` with new crate names
- [ ] Update `.specs/` references to registries
- [ ] Update `ARCHITECTURE_CHANGE_PLAN.md` with canonical terms
- [ ] Update `SECURITY_AUDIT_TRIO_BINARY_ARCHITECTURE.md` with new names
- [ ] Search/replace in all `.md` files: `service-registry` → `node-registry`
- [ ] Search/replace in all `.md` files: `pool-registry-types` → `cluster-types`

### Phase 3: Update Tests (1 hour)

- [ ] Update BDD feature files with new names
- [ ] Update test assertions referencing old crate names
- [ ] Run `cargo test --workspace` to verify

### Phase 4: Update CI/Tooling (30 min)

- [ ] Update `.github/workflows/` with new crate paths
- [ ] Update `xtask` references if any
- [ ] Update `CODEOWNERS` if affected

### Phase 5: Git History Preservation (30 min)

- [ ] Use `git mv` to preserve history:

  ```bash
  git mv bin/pool-managerd-crates/both_registers_question_mark/service-registry \
         bin/pool-managerd-crates/both_registers_question_mark/node-registry
  
  git mv bin/pool-managerd-crates/both_registers_question_mark/node-registration \
         bin/pool-managerd-crates/both_registers_question_mark/node-registration-client
  
  git mv libs/pool-registry-types libs/cluster-types
  ```

---

## PART N: Glossary (Quick Reference)

| Term | Definition | Example |
|------|------------|---------|
| **Single-Node Mode** | 1 machine deployment | Workstation with 2 GPUs |
| **Multi-Node Mode** | 2+ machine deployment | Home lab with 3 GPU servers |
| **Cluster** | Multi-node deployment | Production with 3 GPU nodes |
| **Node** | One physical machine | `gpu-node-1.local` |
| **GPU Device** | One NVIDIA GPU card | `CUDA_VISIBLE_DEVICES=0` |
| **Worker** | One worker-orcd process | `worker-orcd --gpu=0` |
| **Pool** | One model on one GPU | `pool-0` (Llama-3.1-8B) |
| **Replica** | Execution slot in pool | 4 replicas = 4 parallel jobs |
| **Shard** | TP model piece | Llama-70B shard 0/4 |
| **Control Plane** | orchestratord + routing | Admission, placement, streaming |
| **Data Plane** | pool-managerd + workers | Model execution, VRAM |
| **MCD** | Model capability descriptor | Requires `rope_llama`, `gqa` |
| **ECP** | Engine capability profile | Supports `rope_llama`, `mha`, `gqa` |
| **ModelShardHandle** | Sealed VRAM shard | Attestation that weights are resident |
| **NCCL Group** | TP communicator | Workers coordinate via NCCL |

---

## PART O: Final Recommendation

### Proposed Changes (CRITICAL)

1. **Rename `service-registry` → `node-registry`**
2. **Rename `node-registration` → `node-registration-client`**
3. **Rename `pool-registry-types` → `cluster-types`**
4. **Adopt canonical terminology** from this document

### Timeline

- **Review period**: 1 day (2025-10-01)
- **Implementation**: 1 day (2025-10-02)
- **Verification**: Run full test suite
- **Merge**: Before worker-orcd implementation begins

### Rationale

We are **pre-v1.0** and destructive changes are **explicitly allowed** per `.windsurf/rules/destructive-actions.md`. This is the **ideal time** to fix naming confusion before the architecture ossifies.

**Once worker-orcd is implemented, renaming becomes 10x harder.**

---

## Status

**AWAITING APPROVAL** — Please review and respond with:

- ✅ Approve Option A (Node/Pool Distinction)
- 🔄 Request modifications
- ❌ Reject proposal

---

**End of Proposal**
