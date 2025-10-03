# Llama-Orch SPEC — System Architecture

**Status**: Draft  
**Version**: 0.1.0  
**Conformance language**: RFC-2119 (MUST/SHOULD/MAY)

---

## Table of Contents

### 0. Document Metadata
- [0.1 Glossary](#01-glossary)
- [0.2 Traceability Index](#02-traceability-index)

### 1. Executive Summary
- [1.1 Purpose](#11-purpose)
- [1.2 System Architecture](#12-system-architecture)
- [1.3 Decision Hierarchy](#13-decision-hierarchy)

### 2. Foundational Concepts
- [2.1 Model Reference Format](#21-model-reference-format-sys-21x)
- [2.2 VRAM-Only Policy](#22-vram-only-policy-sys-22x)
- [2.3 Determinism Principles](#23-determinism-principles-sys-23x)
- [2.4 Process Isolation Rationale](#24-process-isolation-rationale-sys-24x)
- [2.5 FFI Boundaries](#25-ffi-boundaries-sys-25x)

### 3. Deployment Modes
- [3.1 Home Mode (M0)](#31-home-mode-m0-sys-31x)
- [3.2 Lab Mode (M1)](#32-lab-mode-m1-sys-32x)
- [3.3 Multi-GPU Mode (M1)](#33-multi-gpu-mode-m1-sys-33x)
- [3.4 Multi-Node Mode (M2+)](#34-multi-node-mode-m2-sys-34x)
- [3.5 Platform Mode (Marketplace)](#35-platform-mode-marketplace-sys-35x)

### 4. System-Level Requirements
- [4.1 Intelligence Boundary](#41-intelligence-boundary-sys-41x)
- [4.2 Smart vs Dumb Architecture](#42-smart-vs-dumb-architecture-sys-42x)
- [4.3 Component Separation](#43-component-separation-sys-43x)
- [4.4 State Propagation](#44-state-propagation-sys-44x)
- [4.5 Multi-Node Support](#45-multi-node-support-sys-45x)

### 5. API Contracts
- [5.1 Client → Orchestrator (Agentic API)](#51-client--orchestrator-agentic-api-sys-51x)
- [5.2 Orchestrator ↔ Pool Manager](#52-orchestrator--pool-manager-sys-52x)
- [5.3 Pool Manager ↔ Worker](#53-pool-manager--worker-sys-53x)
- [5.4 Orchestrator → Worker (Direct)](#54-orchestrator--worker-direct-sys-54x)
- [5.5 Error Response Format](#55-error-response-format-sys-55x)
- [5.6 Correlation ID Propagation](#56-correlation-id-propagation-sys-56x)

### 6. Component Architecture
- [6.1 Orchestratord (The Brain)](#61-orchestratord-the-brain-sys-61x)
  - [6.1.1 Orchestrator Intelligence](#611-orchestrator-intelligence-sys-611)
  - [6.1.2 State Management](#612-state-management-sys-612)
  - [6.1.3 Persistent State Store](#613-persistent-state-store-sys-613)
  - [6.1.4 Queue Optimizer](#614-queue-optimizer-sys-614)
  - [6.1.5 Programmable Scheduler](#615-programmable-scheduler-sys-615)
  - [6.1.6 Retry & Backoff Policy](#616-retry--backoff-policy-sys-616)
- [6.2 Pool-Managerd (Control Plane)](#62-pool-managerd-control-plane-sys-62x)
  - [6.2.1 Pool Manager Execution](#621-pool-manager-execution-sys-621)
  - [6.2.2 State Reporting](#622-state-reporting-sys-622)
  - [6.2.3 Preflight Validation](#623-preflight-validation-sys-623)
  - [6.2.4 Heartbeat Protocol](#624-heartbeat-protocol-sys-624)
  - [6.2.5 Operational Cleanup](#625-operational-cleanup-sys-625)
- [6.3 Worker-Orcd (Executor)](#63-worker-orcd-executor-sys-63x)
  - [6.3.1 Worker Self-Containment](#631-worker-self-containment-sys-631)
  - [6.3.2 Worker Isolation](#632-worker-isolation-sys-632)
  - [6.3.3 Tensor Parallelism Design](#633-tensor-parallelism-design-sys-633)
  - [6.3.4 Ready Callback Contract](#634-ready-callback-contract-sys-634)
  - [6.3.5 Cancellation Handling](#635-cancellation-handling-sys-635)

### 7. Data Flow & Interactions
- [7.1 Job Submission Flow](#71-job-submission-flow-sys-71x)
- [7.2 Worker Startup Flow](#72-worker-startup-flow-sys-72x)
- [7.3 Worker Failure Flow](#73-worker-failure-flow-sys-73x)
- [7.4 Cancellation Flow](#74-cancellation-flow-sys-74x)
- [7.5 SSE Reconnection Flow](#75-sse-reconnection-flow-sys-75x)

### 8. Quality Attributes
- [8.1 Determinism](#81-determinism-sys-81x)
- [8.2 Performance](#82-performance-sys-82x)
- [8.3 Reliability](#83-reliability-sys-83x)
- [8.4 Scalability](#84-scalability-sys-84x)

### 9. Security & Compliance
- [9.1 Authentication](#91-authentication-sys-91x)
- [9.2 EU Compliance (GDPR)](#92-eu-compliance-gdpr-sys-92x)
- [9.3 Multi-Tenancy (Platform Mode)](#93-multi-tenancy-platform-mode-sys-93x)

### 10. Metrics & Observability
- [10.1 Metrics Contract](#101-metrics-contract-sys-101x)
- [10.2 Logging Standards](#102-logging-standards-sys-102x)
- [10.3 Correlation & Tracing](#103-correlation--tracing-sys-103x)
- [10.4 Proof Bundle Requirements](#104-proof-bundle-requirements-sys-104x)

### 11. Configuration
- [11.1 Orchestrator Config](#111-orchestrator-config-sys-111x)
- [11.2 Pool Manager Config](#112-pool-manager-config-sys-112x)
- [11.3 Worker Config](#113-worker-config-sys-113x)
- [11.4 Configuration Precedence](#114-configuration-precedence-sys-114x)

### 12. Development Workflow
- [12.1 Spec-Driven Development](#121-spec-driven-development-sys-121x)
- [12.2 Testing Strategy](#122-testing-strategy-sys-122x)
- [12.3 CI/CD Pipeline](#123-cicd-pipeline-sys-123x)

### 13. Crate Dependency Graph
- [13.1 Dependency Overview](#131-dependency-overview)

### 14. Milestone Roadmap
- [14.1 M0: Single GPU (v0.1.0)](#141-m0-single-gpu-v010)
- [14.2 M1: Multi-GPU (v0.2.0)](#142-m1-multi-gpu-v020)
- [14.3 M2: Multi-Node (v0.3.0)](#143-m2-multi-node-v030)
- [14.4 M3: Platform (v0.4.0)](#144-m3-platform-v040)

### 15. Non-Goals / Out of Scope

### 16. References
- [16.1 Specifications](#161-specifications)
- [16.2 Documentation](#162-documentation)
- [16.3 Contracts](#163-contracts)

### 17. Appendices
- [17.1 Resolved Clarifications](#171-resolved-clarifications)
- [17.2 Decision Log](#172-decision-log)
- [17.3 Traceability Matrix](#173-traceability-matrix)

---

## 0. Document Metadata

### 0.1 Glossary

**Key Terms:**

- **Token Budget**: Per-tenant quota for maximum tokens that can be generated daily or per request; enforced at admission time to prevent quota exhaustion.
- **Eviction**: Two types: (1) Model eviction - removing cached models from RAM when no longer needed; (2) Worker eviction - stopping workers to free VRAM for higher-priority jobs.
- **Preflight Validation**: Pre-spawn checks performed by pool-managerd to verify GPU has sufficient free VRAM and model is compatible before spawning a worker.
- **VRAM-Only Policy**: Requirement that all model weights, KV cache, activations, and intermediate tensors reside entirely in GPU VRAM with no RAM/disk fallback.
- **Determinism**: Property where same model + same seed + same prompt produces identical output; system-level guarantee with model-level limitations.
- **Proof Bundle**: Standardized test artifact directory containing seeds, transcripts, metadata, and outputs for reproducibility (see `libs/proof-bundle`).
- **Smart/Dumb Boundary**: Architectural principle where orchestratord makes ALL intelligent decisions (smart) while pool-managerd and workers execute commands without policy decisions (dumb).
- **Model Reference (model_ref)**: Canonical identifier for a model artifact, format: `hf:{org}/{repo}@{rev}::file={path}` or `file:/path/to/model.gguf`.
- **Heartbeat**: Periodic state report from pool-managerd to orchestratord (default 15s interval) containing GPU VRAM state and worker status.
- **SSE (Server-Sent Events)**: HTTP streaming protocol used for token-by-token inference results from worker → orchestrator → client.
- **Correlation ID**: Unique identifier (`X-Correlation-Id` header) propagated across all service calls for request tracing and log correlation.
- **Tenant**: Isolated customer/user in platform mode with separate quotas, billing, and resource allocation.
- **Priority Classes**: Job queue priorities - `interactive` (user-facing, low latency) and `batch` (background, high throughput).

### 0.2 Traceability Index

**Quick Lookup Table:**

| ID Range | Section | Description |
|----------|---------|-------------|
| SYS-2.1.x | Foundational Concepts | Model Reference Format |
| SYS-2.2.x | Foundational Concepts | VRAM-Only Policy |
| SYS-2.3.x | Foundational Concepts | Determinism Principles |
| SYS-2.4.x | Foundational Concepts | Process Isolation |
| SYS-2.5.x | Foundational Concepts | FFI Boundaries |
| SYS-3.1.x | Deployment Modes | Home Mode |
| SYS-3.2.x | Deployment Modes | Lab Mode |
| SYS-3.3.x | Deployment Modes | Multi-GPU Mode |
| SYS-3.4.x | Deployment Modes | Multi-Node Mode |
| SYS-3.5.x | Deployment Modes | Platform Mode |
| SYS-4.1.x | System Requirements | Intelligence Boundary |
| SYS-4.2.x | System Requirements | Smart vs Dumb |
| SYS-4.3.x | System Requirements | Component Separation |
| SYS-4.4.x | System Requirements | State Propagation |
| SYS-4.5.x | System Requirements | Multi-Node Support |
| SYS-5.1.x | API Contracts | Client → Orchestrator |
| SYS-5.2.x | API Contracts | Orchestrator ↔ Pool |
| SYS-5.3.x | API Contracts | Pool ↔ Worker |
| SYS-5.4.x | API Contracts | Orchestrator → Worker |
| SYS-5.5.x | API Contracts | Error Responses |
| SYS-5.6.x | API Contracts | Correlation IDs |
| SYS-6.1.x | Components | Orchestratord |
| SYS-6.2.x | Components | Pool-Managerd |
| SYS-6.3.x | Components | Worker-Orcd |
| SYS-7.1.x | Data Flows | Job Submission |
| SYS-7.2.x | Data Flows | Worker Startup |
| SYS-7.3.x | Data Flows | Worker Failure |
| SYS-7.4.x | Data Flows | Cancellation |
| SYS-7.5.x | Data Flows | SSE Reconnection |
| SYS-8.1.x | Quality | Determinism |
| SYS-8.2.x | Quality | Performance |
| SYS-8.3.x | Quality | Reliability |
| SYS-8.4.x | Quality | Scalability |
| SYS-9.1.x | Security | Authentication |
| SYS-9.2.x | Security | EU Compliance |
| SYS-9.3.x | Security | Multi-Tenancy |
| SYS-10.1.x | Observability | Metrics |
| SYS-10.2.x | Observability | Logging |
| SYS-10.3.x | Observability | Tracing |
| SYS-10.4.x | Observability | Proof Bundles |
| SYS-11.1.x | Configuration | Orchestrator |
| SYS-11.2.x | Configuration | Pool Manager |
| SYS-11.3.x | Configuration | Worker |
| SYS-11.4.x | Configuration | Precedence |
| SYS-12.1.x | Development | Spec-Driven |
| SYS-12.2.x | Development | Testing |
| SYS-12.3.x | Development | CI/CD |

---

## 1. Executive Summary

### 1.1 Purpose

Llama-Orch is a **deterministic, VRAM-only, multi-node GPU orchestration system** for large language model inference. It provides guaranteed reproducibility, EU-native compliance, and enterprise-grade orchestration across distributed GPU resources.

**Core Value Propositions:**
1. **Determinism Guarantee**: Same seed → Same output (every time, provably)
2. **VRAM-Only Policy**: Model fully resident in GPU VRAM (no RAM fallback)
3. **Multi-Node Orchestration**: Distribute models across GPU clusters
4. **EU Compliance**: GDPR-native, EU-only data residency
5. **Marketplace Ready**: Enable GPU provider ecosystem

### 1.2 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ ORCHESTRATORD (The Brain - All Intelligent Decisions)           │
│                                                                   │
│ • Admission, Queue, Scheduling, Worker Selection                 │
│ • Eviction, Retry, Timeout, Cancellation Policies               │
│ • Client-facing API (Platform & Agentic)                        │
│ • SSE Streaming Relay                                           │
└────────────────────┬────────────────────────────────────────────┘
                     │ Commands (HTTP)
                     ├──────────────┬──────────────┬──────────────
                     ↓              ↓              ↓
┌────────────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ POOL-MANAGERD         │ │ POOL-MANAGERD   │ │ POOL-MANAGERD   │
│ (Control Plane)       │ │ (Control Plane) │ │ (Control Plane) │
│                       │ │                 │ │                 │
│ • GPU Inventory       │ │                 │ │                 │
│ • Capability Match    │ │                 │ │                 │
│ • Model Cache         │ │                 │ │                 │
│ • Worker Lifecycle    │ │                 │ │                 │
│ • Operational Cleanup │ │                 │ │                 │
│ • Report State Up     │ │                 │ │                 │
└─────┬──────────────────┘ └─────┬───────────┘ └─────┬───────────┘
      │ Spawns                   │                   │
      ├──────┬──────             ├──────             ├──────
      ↓      ↓      ↓            ↓      ↓            ↓      ↓
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│WORKER│ │WORKER│ │WORKER│ │WORKER│ │WORKER│ │WORKER│ │WORKER│
│      │ │      │ │      │ │      │ │      │ │      │ │      │
│GPU 0 │ │GPU 1 │ │GPU 2 │ │GPU 0 │ │GPU 1 │ │GPU 0 │ │GPU 1 │
│VRAM  │ │VRAM  │ │VRAM  │ │VRAM  │ │VRAM  │ │VRAM  │ │VRAM  │
│Alloc │ │Alloc │ │Alloc │ │Alloc │ │Alloc │ │Alloc │ │Alloc │
└──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘ └──────┘
  Pool 1 GPUs           Pool 2 GPUs           Pool 3 GPUs
```

### 1.3 Decision Hierarchy

```
Orchestratord (Brain) → Makes ALL intelligent decisions
    ↓ commands
Pool Manager (Levers) → Executes commands, reports state
    ↓ spawns
Worker (Executor) → Loads model, executes inference
```

---

## 2. Foundational Concepts

### 2.1 Model Reference Format (SYS-2.1.x)

#### [SYS-2.1.1] Model Reference Accepted Forms

**Accepted forms**:
- Client input (`model` in Agentic API) MAY be one of:
  - `hf:{org}/{repo}` or `hf:{org}/{repo}@{rev}` or `hf:{org}/{repo}@{rev}::file={path}`
  - `file:/abs/path/to/model.gguf`
  - Alias without a scheme (e.g., `llama-7b`)

#### [SYS-2.1.2] Model Reference Resolution

**Resolution rules**:
- Orchestratord (catalog) MUST resolve aliases to a canonical `model_ref` prior to scheduling.
- Orchestratord MUST pin `@rev` to an immutable commit SHA and MUST pin a concrete artifact via `::file=...` for determinism guarantee (strengthened from SHOULD to align with SYS-2.3.1).
- Pool-managerd MUST receive a normalized `model_ref` that starts with `hf:` or `file:` and MUST NOT perform alias resolution.
- The model-provisioner in pool-managerd MUST support:
  - `hf:` — Downloading the specified artifact from Hugging Face.
  - `file:` — Treating the path as a local file.
- Other schemes (e.g., `https:`, `s3:`) MAY be added in future milestones and are NOT REQUIRED for M0.

#### [SYS-2.1.3] Worker Model Contract

**Worker contract**:
- Worker-orcd MUST be provided a concrete filesystem path to the model artifact at startup.
- Worker ready callbacks MUST include the resolved `model_ref` for traceability.

---

### 2.2 VRAM-Only Policy (SYS-2.2.x)

#### [SYS-2.2.1] VRAM-Only Enforcement

The system MUST enforce VRAM-only policy: model weights, KV cache, activations, and intermediate tensors MUST reside entirely in GPU VRAM.

**Prohibited:**
- ❌ RAM fallback
- ❌ Unified memory (CUDA UMA)
- ❌ Zero-copy mode
- ❌ CPU inference fallback
- ❌ Disk swapping

**Rationale**: VRAM-only policy ensures predictable performance and enables deterministic inference by eliminating memory hierarchy variability.

---

### 2.3 Determinism Principles (SYS-2.3.x)

#### [SYS-2.3.1] Determinism Guarantee

The system MUST guarantee deterministic inference: same model + same seed + same prompt → same output.

**Requirements:**
- Sealed VRAM shards (worker-orcd)
- Pinned engine versions
- Deterministic sampling
- No non-deterministic operations

**Design Principle**: The system is designed AS IF deterministic, but acknowledges that underlying models may not guarantee determinism.

#### [SYS-2.3.2] System-Level Guarantees

**System-level guarantees** (what we control):
- Worker-orcd MUST allocate and keep all model weights, KV cache, and activations in VRAM only
- Engine versions and kernel parameters MUST be pinned and recorded per job
- SSE event ordering MUST be stable and reproducible
- Same seed + same inputs MUST follow identical code paths through the system

#### [SYS-2.3.3] Model-Level Limitations

**Model-level limitations** (what we cannot control):
- Inference engines (llama.cpp, vLLM, etc.) MAY have non-deterministic operations
- GPU hardware variations MAY produce different floating-point results
- Model architectures MAY include inherently non-deterministic components
- Cross-worker/cross-GPU determinism is NOT guaranteed

#### [SYS-2.3.4] Best-Effort Determinism

**Best-effort determinism**:
- When model and engine support determinism, system MUST preserve it
- Sampling SHOULD be deterministic for identical inputs and seeds where engine allows
- Non-deterministic operations SHOULD be disabled or replaced when possible
- System MUST document which models/engines have been verified as deterministic

#### [SYS-2.3.5] Recording for Reproducibility

**Recording for reproducibility**:
- Proof bundles MUST include seed, model_ref (pinned @rev and artifact), engine version, device info
- Failed determinism attempts SHOULD be logged with hardware/engine context
- Property tests SHOULD verify determinism for known-deterministic models
- Research document MUST catalog which models/engines achieve determinism (see `.docs/determinism-research.md`)

---

### 2.4 Process Isolation Rationale (SYS-2.4.x)

#### [SYS-2.4.1] Process Isolation Requirement

Workers MUST run in separate processes from pool managers.

**Why**: CUDA allocations are per-process. Workers need self-contained VRAM ownership within their CUDA context.

#### [SYS-2.4.2] Worker Process Isolation

Workers MUST run in separate processes:

**Requirements**:
- Each worker MUST have its own OS process
- Each worker MUST have its own CUDA context
- Workers MUST NOT share VRAM pointers
- Worker MUST own complete VRAM lifecycle (allocate → use → free)

**Rationale**: CUDA VRAM allocations are per-process. Workers need isolated VRAM ownership.

**Testing benefit**: Enables standalone worker testing (`worker-orcd --model X --gpu 0`).

**Communication**: Components MUST communicate via HTTP APIs only.

---

### 2.5 FFI Boundaries (SYS-2.5.x)

#### [SYS-2.5.1] FFI Boundary Enforcement

The system MUST enforce strict FFI boundaries:

**Pool manager** (NVML only):
- MUST use NVML for read-only GPU queries
- MUST query system-wide state (all GPUs)
- MUST NOT allocate VRAM or use CUDA
- MUST NOT perform compute operations

**Worker** (CUDA only):
- MUST use CUDA Runtime API for VRAM allocation
- MUST allocate VRAM within its process CUDA context
- MUST use CUDA for compute operations
- MUST own VRAM lifecycle (allocate → use → free)

**Rationale**: Pool manager monitors system-wide GPU state. Worker manages per-process VRAM within its CUDA context. These are orthogonal concerns with different FFI layers.

---

## 3. Deployment Modes

The system supports five distinct deployment modes with different security and operational characteristics.

**Mode Selection Principle**:
- **Home Mode**: Performance > Security (all components on same system)
- **Lab Mode**: Balanced (orchestrator separate from pools)
- **Platform Mode**: Security > Performance (multi-tenant marketplace)

### 3.1 Home Mode (M0) (SYS-3.1.x)

#### [SYS-3.1.1] Home Mode Deployment

**Single node, all components co-located**

```
[Orchestrator] (localhost:8080)
      ↓
[Pool Manager] (localhost:9200)
      ↓
[Worker-1] GPU 0 (localhost:8001)
```

**Use case**: Development, home lab, single user, local experimentation

**Characteristics**:
- All 3 binaries on same system
- No authentication required (localhost trust)
- No audit logging overhead
- Performance optimized
- Minimal configuration

**Requirements**:
- Orchestratord and pool-managerd MUST bind to loopback (127.0.0.1) by default.
- Authentication MUST be disabled by default for localhost communication.
- A single worker per GPU MUST be enforced (batch=1) and VRAM-only policy MUST be enabled.
- Configuration SHOULD be minimal; sensible defaults MUST be provided for ports and timeouts.

---

### 3.2 Lab Mode (M1) (SYS-3.2.x)

#### [SYS-3.2.1] Lab Mode Deployment

**Orchestrator on separate machine from GPU pools**

```
[Orchestrator] (lab-controller:8080)
      ↓ (network)
[Pool Manager] (gpu-node-1:9200)
      ├─→ [Worker-1] GPU 0
      └─→ [Worker-2] GPU 1
```

**Use case**: Research lab, small team, distributed GPU resources

**Characteristics**:
- Orchestrator and pool-managerd on different systems
- Authentication enabled by default (network communication)
- Audit logging for inter-service calls
- Balanced security and performance
- Supports multiple pool managers

**Requirements**:
- Authentication MUST be enabled for orchestrator ↔ pool-managerd communication.
- Pool-managerd MUST authenticate registration and heartbeat requests using bearer tokens or mTLS.
- Orchestrator SHOULD bind to network interface (not loopback) with explicit configuration.
- Network communication SHOULD use TLS or run on trusted network segments.

---

### 3.3 Multi-GPU Mode (M1) (SYS-3.3.x)

#### [SYS-3.3.1] Multi-GPU Mode Deployment

**Single node, multiple GPUs**

```
[Orchestrator] (localhost:8080)
      ↓
[Pool Manager] (localhost:9200)
      ├─→ [Worker-1] GPU 0 (localhost:8001)
      ├─→ [Worker-2] GPU 1 (localhost:8002)
      └─→ [Worker-3] GPU 2 (localhost:8003)
```

**Use case**: Single powerful machine with 2-4 GPUs

**Features**:
- Tensor parallelism (split large models across GPUs)
- Multiple models loaded simultaneously

**Requirements**:
- Orchestratord MUST remain single-instance with persistent state (see SYS-6.1.3); restarts MUST reload queue state and allow SSE reconnection via checkpoints.
- Pool-managerd MUST manage workers: single-GPU workers map 1:1 to GPUs; multi-GPU workers (tensor parallel) map 1:N and are tracked with device masks.
- VRAM accounting MUST handle per-GPU allocation and reserved headroom; preflight MUST fail when insufficient free VRAM across required GPUs.
- Cancellation and timeouts MUST apply per-job; multi-GPU workers MUST release all GPU allocations on job termination.

---

### 3.4 Multi-Node Mode (M2+) (SYS-3.4.x)

#### [SYS-3.4.1] Multi-Node Mode Deployment

**Multiple nodes, multiple GPUs each**

```
[Orchestrator] (orchestrator.local:8080)
      ├─→ [Pool Manager 1] (node1:9200)
      │        ├─→ [Worker-1] GPU 0
      │        └─→ [Worker-2] GPU 1
      ├─→ [Pool Manager 2] (node2:9200)
      │        ├─→ [Worker-3] GPU 0
      │        └─→ [Worker-4] GPU 1
      └─→ [Pool Manager 3] (node3:9200)
               ├─→ [Worker-5] GPU 0
               └─→ [Worker-6] GPU 1
```

**Use case**: Enterprise, data center, GPU cluster

**Features**:
- Cluster-wide orchestration
- Load balancing across nodes
- High availability

**Requirements**:
- Orchestratord MUST manage multiple pool-managerd across nodes and MUST make all placement decisions centrally (no nested orchestrators).
- Heartbeat aggregation MUST tolerate partial failures; pools missing heartbeats beyond timeout MUST be excluded from scheduling.
- Network communication between nodes SHOULD use TLS or a trusted network; service discovery MAY be static or registry-based.
- Scheduling SHOULD prefer locality and capacity signals (e.g., free VRAM); policy MUST be observable via metrics and logs.

---

### 3.5 Platform Mode (Marketplace) (SYS-3.5.x)

#### [SYS-3.5.1] Platform Mode Deployment

**Federation across provider orchestrators**

```
[Platform Orchestrator] (api.yourplatform.com:443)
      ├─→ [Provider A Orchestrator] (provider-a.internal:8080)
      │        └─→ Provider A's pools/workers
      ├─→ [Provider B Orchestrator] (provider-b.internal:8080)
      │        └─→ Provider B's pools/workers
      └─→ [Provider C Orchestrator] (provider-c.internal:8080)
               └─→ Provider C's pools/workers
```

**Use case**: GPU marketplace, provider ecosystem

**Features**:
- Provider registration
- Federated routing (NOT nesting)
- Billing and usage tracking
- Multi-tenancy and quotas

**Key distinction**: Platform orchestrator is a **smart router** that selects providers based on capacity/cost/region, not a nested orchestrator. Provider orchestrators make their own worker placement decisions within their pools.

**Business doc**: `.docs/.business/monetization.md`

**Requirements**:
- Platform orchestrator MUST authenticate providers and MUST maintain a registry of provider endpoints and capabilities.
- Routing MUST be federated (provider orchestrators make placement decisions); nested scheduling MUST NOT occur.
- Tenancy context MUST be preserved across federation and included in billing records.
- Provider-level SLIs/labels (availability, capacity) SHOULD inform routing; fallbacks MAY be used when signals are missing.

---

## 4. System-Level Requirements

### 4.1 Intelligence Boundary (SYS-4.1.x)

#### [SYS-4.1.1] Intelligence Centralization

The system MUST centralize ALL intelligent decisions in orchestratord. Pool managers and workers MUST be dumb executors.

**Intelligent decisions** (orchestratord only):
- Admission (accept/reject requests)
- Scheduling (which job next)
- Worker selection (where to run)
- Eviction (which worker to stop)
- Retry (whether to retry failures)
- Timeout enforcement
- Cancellation propagation

**Dumb execution** (pool managers and workers):
- Execute commands received
- Report facts and state
- Operational cleanup (no policy decisions)

---

### 4.2 Smart vs Dumb Architecture (SYS-4.2.x)

#### [SYS-4.2.1] Smart vs Dumb Boundary

The system MUST enforce a strict smart/dumb boundary:

**Smart components** (orchestratord only):
- MUST make ALL policy decisions (admission, scheduling, eviction, retry, timeout)
- MUST use configured policies (no hardcoded decisions)
- MUST decide actions based on aggregated state

**Dumb components** (pool manager, worker):
- MUST execute commands received without interpretation
- MUST report facts and state without filtering
- MUST NOT make policy decisions
- MUST perform operational cleanup only (not recovery decisions)

---

### 4.3 Component Separation (SYS-4.3.x)

#### [SYS-4.3.1] Binary Separation

The system MUST implement three separate binaries: orchestratord, pool-managerd, and worker-orcd. Each component MUST communicate via HTTP APIs only.

#### [SYS-4.3.2] No Direct Client→Worker Communication

Clients MUST NOT communicate directly with workers. All client requests MUST go through orchestratord.

**Clarification**: The orchestrator directly calls worker endpoints to proxy/relay requests, but clients never communicate with workers directly. This maintains the control plane boundary.

---

### 4.4 State Propagation (SYS-4.4.x)

#### [SYS-4.4.1] Unidirectional State Flow

The system MUST implement unidirectional state flow:

**Upward state flow** (MUST report facts):
```
Worker → Pool Manager → Orchestrator
  (VRAM usage)  (GPU state)  (Cluster state)
```
- Worker MUST report VRAM usage to pool manager
- Pool manager MUST aggregate GPU state and report to orchestrator
- Orchestrator MUST query pool managers for scheduling decisions

**Downward command flow** (MUST execute commands):
```
Orchestrator → Pool Manager → Worker
  (decisions)   (execution)   (inference)
```
- Orchestrator MUST send commands (start worker, dispatch job)
- Pool manager MUST execute commands (spawn worker)
- Worker MUST execute inference requests

---

### 4.5 Multi-Node Support (SYS-4.5.x)

#### [SYS-4.5.1] Distributed Deployment Support

The system MUST support distributed deployment across multiple GPU nodes.

**Deployment modes:**
- Single node, single GPU (M0)
- Single node, multi GPU (M1)
- Multi node, multi GPU (M2+)

**Requirements:**
- Orchestratord MUST support registration of multiple pool-managerd instances across nodes
- Network communication MUST use HTTP APIs with optional TLS
- Heartbeat protocol MUST tolerate network latency and partial failures
- Scheduling decisions MUST account for cross-node placement costs

---

## 5. API Contracts

This section defines all HTTP API contracts between components. APIs MUST be versioned (e.g., `/v2/`) and backward-compatible within major versions.

### 5.1 Client → Orchestrator (Agentic API) (SYS-5.1.x)

#### [SYS-5.1.1] Task Submission

**Task submission**:
```
POST /v2/tasks
{
  "session_id": "sess-abc",
  "model": "llama-3.1-8b",
  "prompt": "Hello world",
  "max_tokens": 100,
  "temperature": 0.7,
  "seed": 42,
  "priority": "interactive"
}

Response (202 Accepted):
{
  "job_id": "job-xyz",
  "status": "queued",
  "queue_position": 2,
  "events_url": "/v2/tasks/job-xyz/events"
}
```

#### [SYS-5.1.2] Task Submission Requirements

**Requirements**:
- Requests MUST include `model`, `prompt`, and `max_tokens`
- `temperature` is OPTIONAL with default 0.7 for sampling; deterministic mode uses seed without temperature
- `seed` SHOULD be provided for determinism, otherwise orchestrator MAY supply one and MUST record it
- `priority` MUST be one of `interactive` or `batch`; unknown values MUST be rejected with 400
- Orchestratord MUST validate model existence and token budgets at admission; invalid requests MUST return 4xx with a stable error code
- When the queue is full and policy=`reject`, orchestrator MUST return 429 with a Retry-After where possible; for `drop-lru`, orchestrator MUST emit metrics indicating eviction

#### [SYS-5.1.3] SSE Streaming

**Streaming**:
```
GET /v2/tasks/{job_id}/events (SSE)

Events:
- queued → started → token* → metrics* → end
- error (if failure or cancellation)
```

**Requirements**:
- SSE endpoint MUST be idempotent to reconnects for the same `job_id`
- Orchestrator SHOULD resume streaming from the last sent offset using `sse_checkpoints` table (see SYS-6.1.3)
- Clients MAY provide `Last-Event-ID` header to indicate last received event
- Event order MUST be: `queued` → `started` → zero or more `token` → zero or more `metrics` (interleaved) → terminal (`end` or `error`)
- Exactly one terminal event MUST be emitted per job

**Specs**: `bin/orchestratord-crates/agentic-api/.specs/00_agentic_api.md`

---

### 5.2 Orchestrator ↔ Pool Manager (SYS-5.2.x)

#### [SYS-5.2.1] Pool Registration

**Pool registration**:
```
POST /v2/pools/register
{
  "pool_id": "pool-1",
  "endpoint": "http://192.168.1.100:9200",
  "gpus": [...]
}
```

**Requirements**:
- Pool registration MUST be authenticated (see SYS-9.1.x) and MUST include `pool_id`, `endpoint`, and initial GPU inventory
- Orchestratord MUST validate `pool_id` uniqueness and SHOULD reject re-registration attempts that conflict without explicit takeover semantics

#### [SYS-5.2.2] Heartbeat Protocol

**Heartbeat**:
```
POST /v2/pools/{id}/heartbeat
{
  "pool_id": "pool-1",
  "gpus": [...],
  "workers": [...]
}
```

**Requirements**:
- Heartbeats MUST be sent at the configured interval (default 15s) and MUST include current GPU VRAM state and worker states
- Missing heartbeats beyond timeout MUST mark the pool unavailable for scheduling
- Heartbeat payload MUST include timestamp for clock skew detection

#### [SYS-5.2.3] Worker Start Command

**Worker start command**:
```
POST /v2/workers/start
{
  "model_ref": "hf:author/repo@rev::file=models/model.Q4_K_M.gguf",
  "gpu_id": 0
}
```

**Requirements**:
- Worker start commands MUST be accepted only after preflight checks (VRAM availability and compatibility) have passed in pool-managerd
- Response MUST include worker_id assigned by pool-managerd
- Failures MUST return stable error codes (e.g., `INSUFFICIENT_VRAM`, `MODEL_INCOMPATIBLE`)

**Specs**: `bin/pool-managerd-crates/control-api/.specs/00_control_api.md`

---

### 5.3 Pool Manager ↔ Worker (SYS-5.3.x)

#### [SYS-5.3.1] Worker Ready Callback

**Worker ready callback**:
```
POST /v2/internal/workers/ready
{
  "worker_id": "worker-abc",
  "model_ref": "llama-7b",
  "vram_bytes": 17000000000,
  "uri": "http://localhost:8001"
}
```

**Requirements**:
- Worker MUST issue ready callback after HTTP server starts (server-first)
- Callback MUST include `worker_id`, `model_ref`, `vram_bytes`, and `uri`
- Pool-managerd MUST update VRAM accounting and mark worker ready atomically
- Callback failures MUST trigger worker cleanup in pool-managerd

**Specs**: `bin/pool-managerd-crates/worker-lifecycle/.specs/00_worker_lifecycle.md`

---

### 5.4 Orchestrator → Worker (Direct) (SYS-5.4.x)

#### [SYS-5.4.1] Inference Execution

**Inference execution**:
```
POST {worker_uri}/execute
{
  "job_id": "job-xyz",
  "prompt": "Hello world",
  "max_tokens": 100,
  "seed": 42
}

Response: SSE stream
- started
- token* (multiple)
- metrics* (periodic)
- end
- error (on failure or cancellation)
```

**Requirements**:
- SSE stream event order MUST be: `started` → zero or more `token` → zero or more `metrics` (interleaved as needed) → terminal (`end` or `error`)
- Exactly one terminal event MUST be emitted per job (`end` on success, `error` on failure/cancel)
- Workers MUST include stable error codes; cancellation MUST use `CANCELLED`
- Orchestrator MUST propagate `X-Correlation-Id` to worker requests; workers SHOULD echo it in SSE metadata
- Orchestrator SHOULD enforce request timeouts per job policy and close SSE cleanly on timeout with a terminal event

#### [SYS-5.4.2] Cancellation

**Cancellation**:
```
POST {worker_uri}/cancel
{
  "job_id": "job-xyz"
}
```

**Cancellation semantics**:
- Idempotent: repeated cancels for the same `job_id` MUST be safe and return the same terminal outcome
- Prompt propagation: orchestrator MUST issue cancel immediately on client request or stream disconnect
- Worker behavior: upon cancel, worker MUST stop decoding promptly, free resources, and emit SSE `error` with stable code `CANCELLED`
- Acknowledgement: worker SHOULD return HTTP 202 for `POST /cancel` if cancellation has been accepted
- Deadline: orchestrator SHOULD enforce a cancellation deadline (default 5s) after which it treats the job as cancelled and closes client SSE

**Specs**: `bin/worker-orcd/.specs/00_worker-orcd.md`

---

### 5.5 Error Response Format (SYS-5.5.x)

#### [SYS-5.5.1] Standard Error Response

All API errors MUST use a standard JSON error response format:

```json
{
  "error": {
    "code": "INSUFFICIENT_VRAM",
    "message": "GPU 0 has 12GB free, but model requires 16GB",
    "details": {
      "gpu_id": 0,
      "available_vram_bytes": 12884901888,
      "required_vram_bytes": 17179869184
    },
    "correlation_id": "req-abc-123"
  }
}
```

**Requirements**:
- `code` MUST be a stable, uppercase identifier (e.g., `INSUFFICIENT_VRAM`, `MODEL_NOT_FOUND`, `QUOTA_EXCEEDED`)
- `message` SHOULD be human-readable and MAY include context
- `details` is OPTIONAL and MAY include structured diagnostic information
- `correlation_id` MUST be included for traceability
- HTTP status codes MUST align with error semantics (400 for client errors, 500 for server errors, 429 for rate limits, 503 for unavailable)

---

### 5.6 Correlation ID Propagation (SYS-5.6.x)

#### [SYS-5.6.1] Correlation ID Requirements

**Correlation**:
- `X-Correlation-Id` MUST be accepted from clients and propagated across orchestrator → pool-managerd → worker calls
- If the header is absent, orchestrator MUST generate a new correlation ID and propagate it downstream
- All logs and error responses MUST include the correlation ID
- SSE events SHOULD include correlation ID in metadata for client-side tracing

---

## 6. Component Architecture

### 6.1 Orchestratord (The Brain) (SYS-6.1.x)

**Binary**: `bin/orchestratord/`  
**Port**: 8080 (default)  
**Role**: Centralized intelligence for all decisions

**Crates:**
- `scheduling` — Admission, queue, job tracking, worker selection, eviction
- `platform-api` — Marketplace federation facade
- `agentic-api` — Standard/home orchestrator API
- `pool-registry` — Track pool managers and state
- `streaming` — SSE relay with metadata
- `task-cancellation` — Cancellation propagation
- `job-timeout` — Timeout enforcement
- `backpressure` — Queue backpressure handling
- `state-store` — Persistent state management (see SYS-6.1.3)
- `queue-optimizer` — Background optimization cron job (see SYS-6.1.4)

**Specs**: `bin/orchestratord/.specs/00_orchestratord.md`

---

#### [SYS-6.1.1] Orchestrator Intelligence

Orchestratord MUST implement ALL intelligent decision-making:
- MUST validate and admit requests before enqueue
- MUST manage bounded FIFO queue with Interactive/Batch priorities
- MUST select next job and target worker (combined scheduling decision)
- MUST command pool managers to start/stop workers
- MUST route inference requests to selected workers
- MUST relay SSE streams from workers to clients
- MUST enforce timeout limits on jobs
- MUST propagate cancellation requests to workers

**Requirements**:
- All intelligent decisions (admission, scheduling, eviction, retry, timeout, cancellation) MUST occur in orchestratord and MUST NOT be delegated
- SSE relay to clients MUST preserve worker event order and MUST add orchestrator metadata without altering payload semantics

---

#### [SYS-6.1.2] State Management

Orchestratord MUST maintain persistent state for job tracking, queue management, and operational history to enable intelligent scheduling decisions, graceful restarts, and client reconnection.

**State categories**:

**Ephemeral state** (in-memory, lost on restart):
- Active SSE connections and stream buffers
- Cached pool manager states (refreshed via heartbeat)
- In-flight HTTP requests to workers
- Performance metrics aggregates (exported to Prometheus)

**Persistent state** (durable, survives restart):
- Job records (id, status, model, params, timestamps, worker assignment)
- Queue state (pending jobs with priority and position)
- Job history (completed/failed jobs with outcomes for audit)
- Tenant quotas and usage accounting
- Configuration and policies

**Requirements**:
- Orchestratord MUST persist job state to enable restart without job loss
- On restart it MUST resume queue processing and allow SSE reconnection via checkpoints

---

#### [SYS-6.1.3] Persistent State Store

Orchestratord MUST use a persistent state store to maintain durable job and queue state across restarts.

**Storage technology selection**:

**RECOMMENDED: SQLite (embedded relational DB)**
- ✅ ACID transactions for job state consistency
- ✅ Zero-ops: no separate database server required
- ✅ File-based: simple backup/restore
- ✅ SQL queries for job history and analytics
- ✅ Write-ahead logging (WAL) for concurrent reads
- ✅ Proven reliability and performance for orchestrator workloads
- ✅ Rust support via `rusqlite` or `sqlx`

**Alternative: PostgreSQL (client-server relational DB)**
- Use case: Multi-orchestrator HA setup (future M2+)
- Requires separate database server and operational overhead
- NOT REQUIRED for M0/M1 (single orchestrator)

**NOT RECOMMENDED**:
- ❌ Document/NoSQL databases (MongoDB, CouchDB): Overkill, adds complexity
- ❌ Key-value stores (Redis, etcd): Lack relational queries for job history
- ❌ JSON files: No ACID, no concurrent access, no query capabilities

**Schema design** (SQLite tables):

```sql
-- Job records with full lifecycle
CREATE TABLE jobs (
  job_id TEXT PRIMARY KEY,
  tenant_id TEXT,
  session_id TEXT,
  status TEXT NOT NULL, -- queued, dispatched, running, completed, failed, cancelled
  priority TEXT NOT NULL, -- interactive, batch
  model_ref TEXT NOT NULL,
  prompt_hash TEXT, -- SHA256 of prompt (not raw prompt for privacy)
  max_tokens INTEGER NOT NULL,
  seed INTEGER,
  worker_id TEXT, -- assigned worker (NULL if not dispatched)
  pool_id TEXT, -- assigned pool (NULL if not dispatched)
  
  created_at INTEGER NOT NULL, -- Unix timestamp ms
  queued_at INTEGER, -- when admitted to queue
  dispatched_at INTEGER, -- when sent to worker
  started_at INTEGER, -- when worker began execution
  completed_at INTEGER, -- when terminal event received
  
  outcome TEXT, -- success, error, timeout, cancelled
  error_code TEXT, -- stable error code if failed
  tokens_generated INTEGER, -- final token count
  
  retry_count INTEGER DEFAULT 0,
  last_error TEXT, -- error message from last failure
  
  INDEX idx_status (status),
  INDEX idx_tenant (tenant_id, created_at),
  INDEX idx_worker (worker_id, status)
);

-- Queue state (denormalized for fast dequeue)
CREATE TABLE queue (
  job_id TEXT PRIMARY KEY,
  priority TEXT NOT NULL,
  queued_at INTEGER NOT NULL,
  tenant_id TEXT,
  
  FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE,
  INDEX idx_priority_time (priority, queued_at)
);

-- SSE stream checkpoints for reconnection
CREATE TABLE sse_checkpoints (
  job_id TEXT PRIMARY KEY,
  last_event_seq INTEGER NOT NULL, -- sequence number of last sent event
  last_event_type TEXT, -- token, metrics, end, error
  updated_at INTEGER NOT NULL,
  
  FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
);

-- Tenant quotas and usage
CREATE TABLE tenant_quotas (
  tenant_id TEXT PRIMARY KEY,
  max_concurrent_jobs INTEGER,
  max_vram_bytes INTEGER,
  token_budget_daily INTEGER,
  tokens_used_today INTEGER DEFAULT 0,
  quota_reset_at INTEGER -- Unix timestamp for daily reset
);

-- Configuration and policies (key-value)
CREATE TABLE config (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at INTEGER NOT NULL
);
```

**Operational requirements**:

- **Durability**: SQLite MUST use WAL mode (`PRAGMA journal_mode=WAL`) for durability and concurrent reads.
- **Transactions**: Job state transitions MUST use transactions to ensure consistency (e.g., dequeue + dispatch is atomic).
- **Cleanup**: Completed/failed jobs older than retention period (default 7 days) SHOULD be archived or purged.
- **Backup**: SQLite file SHOULD be backed up periodically; WAL checkpoint MUST be performed before backup.
- **Migration**: Schema migrations MUST be versioned and applied automatically on orchestrator startup.

**Performance considerations**:

- **Write throughput**: SQLite handles 10k+ writes/sec with WAL mode (sufficient for orchestrator workload).
- **Read concurrency**: WAL mode allows unlimited concurrent readers while writer proceeds.
- **Indexes**: Queries MUST use indexes on `status`, `tenant_id`, `priority`, and `queued_at` for fast scheduling.
- **Connection pooling**: Use connection pool (e.g., `sqlx::Pool`) with 4-8 connections for concurrent access.

**Restart behavior**:

1. On startup, orchestrator MUST:
   - Open SQLite database (create if missing, apply migrations)
   - Load queue state: SELECT jobs with `status='queued'` ORDER BY priority, queued_at
   - Mark stale `dispatched`/`running` jobs as `failed` (worker may have crashed during downtime)
   - Resume queue processing

2. On SSE reconnect:
   - Client provides `job_id` and optional `Last-Event-ID` header
   - Orchestrator queries `sse_checkpoints` table for last sent event
   - If job still active, resume streaming from checkpoint
   - If job completed, send cached terminal event from `jobs` table

**Specs**: `bin/orchestratord-crates/state-store/.specs/00_state_store.md`

---

#### [SYS-6.1.4] Queue Optimizer

Orchestratord MUST run a background optimizer to re-evaluate queue and pool states for potential improvements when queue depth exceeds threshold.

**Purpose**: Optimize job placement and resource utilization when queue builds up.

**Trigger conditions**:
- Queue depth exceeds configurable threshold (default: 10 jobs)
- Optimizer runs periodically (default: every 30 seconds) while threshold exceeded
- Stops when queue depth falls below threshold

**Optimization actions**:
- Re-evaluate job-to-worker assignments based on current pool states
- Identify opportunities for better placement (e.g., jobs waiting for busy workers when idle workers available)
- Suggest worker starts/stops to pool managers (e.g., preload models for queued jobs)
- Reorder queue based on updated priorities or resource availability
- Detect and flag stale jobs (e.g., jobs queued for unavailable models)

**Operational requirements**:
- Optimizer MUST NOT block normal scheduling operations
- Optimizations MUST be advisory; scheduler makes final decisions
- Optimizer SHOULD use read-only queries to state store to avoid lock contention
- Optimization cycle SHOULD complete within 5 seconds to avoid staleness
- Orchestrator MUST emit metrics for optimizer activity (runs, suggestions, improvements)

**Configuration**:
```yaml
queue_optimizer:
  enabled: true
  threshold_jobs: 10  # start optimizer when queue > 10
  interval_ms: 30000  # run every 30s while active
  max_runtime_ms: 5000  # abort if cycle exceeds 5s
```

**Specs**: `bin/orchestratord-crates/queue-optimizer/.specs/00_queue_optimizer.md`

---

#### [SYS-6.1.5] Programmable Scheduler

Orchestratord scheduler is designed as a **policy execution engine** that can run user-defined scheduling logic while maintaining a high-performance default.

**Deployment mode behavior**:
- **Platform Mode**: Uses immutable, built-in scheduler (written in Rhai) optimized for multi-tenant fairness, security, and SLA compliance
- **Home/Lab Mode**: Users can write custom Rhai scripts or YAML configurations to define scheduling policies
- **Web UI Mode**: Visual policy builder generates Rhai or YAML for non-programmers

**Language support**:
- **Rhai** (primary): Rust-native scripting language with type safety, 0-indexed arrays, and built-in sandboxing
- **YAML** (declarative): Compiles to Rhai internally for simple rule-based policies

**Scheduler API**:
- Complete system state access (queue, pools, workers, GPUs, models, tenants)
- 40+ built-in helper functions (worker selection, GPU queries, quota checks, eviction)
- Preloaded model catalog at compile time
- Real-time pool state from heartbeats
- Sandboxed execution with 50ms timeout and memory limits

**Platform scheduler** (reference implementation):
- Location: `bin/orchestratord-crates/scheduling/platform-scheduler.rhai`
- Immutable in platform mode, copyable in home/lab mode
- Optimized for priority-based scheduling, quota enforcement, and resource utilization

**Specs**: 
- `bin/orchestratord-crates/scheduling/.specs/00_programmable_scheduler.md` — Overall design and architecture
- `bin/orchestratord-crates/scheduling/.specs/01_rhai_scheduler_runtime.md` — Rhai runtime environment and API reference

---

#### [SYS-6.1.6] Retry & Backoff Policy

Orchestratord MUST implement configurable retry and backoff policies for failed jobs and worker operations.

**Retry backoff configuration**:
```yaml
retry:
  enabled: true
  max_attempts: 5
  initial_delay_ms: 1000  # 1 second
  multiplier: 2.0  # exponential: 1s, 2s, 4s, 8s, 16s
  max_delay_ms: 60000  # cap at 60 seconds
  jitter: true  # add random jitter to prevent thundering herd
```

**Requirements**:
- Retries MUST apply exponential backoff with configurable parameters
- Backoff MUST be per-job to avoid cascading failures
- Jitter SHOULD be applied to prevent thundering herd effects
- Max retry attempts MUST be enforced; exceeded attempts MUST fail the job with stable error code
- Retry decisions MUST be logged with attempt count and backoff duration
- Retries are for fault tolerance, not determinism guarantees (see SYS-2.3.3)

**Acknowledgement**: Cross-worker retries MAY produce different results due to model-level non-determinism (see SYS-2.3.3).

---

### 6.2 Pool-Managerd (Control Plane) (SYS-6.2.x)

**Binary**: `bin/pool-managerd/`  
**Port**: 9200 (default)  
**Role**: Local agent on GPU nodes, executes orchestrator commands

**Crates:**
- `gpu-inventory` — NVML FFI for GPU/VRAM tracking
- `capability-matcher` — Preflight model compatibility validation
- `model-cache` — Model storage
- `model-provisioner` — Model download orchestration
- `model-catalog` — Model metadata registry
- `worker-lifecycle` — Worker process spawning and monitoring
- `control-api` — HTTP API for orchestrator commands
- `error-ops` — Operational cleanup (not policy)
- `pool-registration-client` — Register with orchestrator/platform

**FFI Boundary**: Uses **NVML** (read-only GPU queries), NOT CUDA

**Specs**: `bin/pool-managerd/.specs/00_pool-managerd.md`

---

#### [SYS-6.2.1] Pool Manager Execution

Pool-managerd MUST execute orchestrator commands without making policy decisions:
- MUST query GPU state via NVML (read-only)
- MUST download and cache models as commanded
- MUST validate model-GPU compatibility before worker start (preflight)
- MUST spawn worker processes as commanded
- MUST update VRAM accounting when workers start/stop
- MUST send periodic heartbeats to orchestratord (default 15s interval)
- MUST perform operational cleanup on worker failures (no retry decisions)

**Requirements**:
- Pool-managerd MUST NOT perform placement or retry decisions; it MUST execute orchestrator commands and report facts only
- NVML MUST be used for GPU queries; CUDA allocations MUST NOT be performed by pool-managerd

---

#### [SYS-6.2.2] State Reporting

Pool-managerd MUST report facts, not decisions:
- MUST report GPU VRAM state (total, available, allocated)
- MUST report worker state (running, ready, failed)
- MUST report failures with context (exit code, error message)
- MUST NOT decide to retry or failover

**Requirements**:
- Heartbeats MUST include GPU VRAM totals/allocated and worker states
- Missed heartbeats beyond timeout MUST mark the pool unavailable for scheduling

---

#### [SYS-6.2.3] Preflight Validation

Pool-managerd MUST perform preflight validation before spawning workers:
- MUST validate GPU has sufficient free VRAM for model
- MUST validate model compatibility with GPU capabilities (compute capability, architecture)
- MUST check model file exists and is readable
- MUST verify no conflicting workers on target GPU(s)

**Requirements**:
- Preflight MUST fail fast with stable error codes (e.g., `INSUFFICIENT_VRAM`, `MODEL_INCOMPATIBLE`)
- Preflight failures MUST NOT spawn worker processes
- VRAM headroom calculation MUST account for system overhead and reserved memory

---

#### [SYS-6.2.4] Heartbeat Protocol

Pool-managerd MUST send periodic heartbeats to orchestratord:
- Default interval: 15 seconds (configurable)
- Payload MUST include: pool_id, timestamp, GPU states, worker states
- Heartbeat MUST be sent even if no state changes occurred

**Requirements**:
- Heartbeat failures MUST be logged but MUST NOT stop the pool manager
- Orchestratord MUST mark pools unavailable after missing N consecutive heartbeats (default: 3)
- Clock skew detection SHOULD be implemented using timestamp comparison

---

#### [SYS-6.2.5] Operational Cleanup

Pool-managerd MUST perform operational cleanup on worker failures:
- MUST remove worker from registry
- MUST release VRAM accounting for failed worker
- MUST kill zombie processes
- MUST close file handles
- MUST report failure to orchestratord with exit code and context

**Requirements**:
- Cleanup MUST be prompt (within 5 seconds of detection)
- Cleanup MUST NOT make retry decisions (orchestratord decides)
- Cleanup failures MUST be logged and MAY trigger pool-level alerts

---

### 6.3 Worker-Orcd (Executor) (SYS-6.3.x)

**Binary**: `bin/worker-orcd/`  
**Port**: Dynamic (assigned by pool manager)  
**Role**: Execute inference on single GPU, or multi-GPU via single process for tensor parallelism

**Crates:**
- **Integrated binary** — All functionality integrated into single binary due to CUDA context requirements
- **CUDA modules** — `src/cuda/` (context, model, inference, health) - process-level operations
- **HTTP handlers** — `src/http/` (execute, health endpoints)
- **Lifecycle** — `src/startup.rs` (initialization and callbacks)

**Binary structure:**
- `src/cuda/` — CUDA FFI layer (memory allocation, kernels, enforcement) - process-level
- `src/main.rs` — Worker entry point and orchestration

**Specs**: `bin/worker-orcd/.specs/00_worker-orcd.md`

---

#### [SYS-6.3.1] Worker Self-Containment

Worker-orcd MUST operate as a self-contained process:
- MUST load exactly ONE model at startup (from disk to VRAM)
- MUST own VRAM allocation within its CUDA context
- MUST allocate all model resources in VRAM only (no RAM fallback)
- MUST execute inference requests received via HTTP
- MUST stream results via SSE (token-by-token)
- MUST monitor VRAM residency (self-health checks)
- MUST report actual VRAM usage to pool manager on ready

**Requirements**:
- Worker-orcd MUST load exactly one model at startup and MUST own all VRAM allocations within its process CUDA context
- All model weights, KV cache, and activations MUST reside in VRAM (no RAM/unified memory fallback)

---

#### [SYS-6.3.2] Worker Isolation

Each worker MUST run in a separate OS process. Workers MUST NOT share VRAM or CUDA contexts.

**Requirements**:
- Each worker MUST have its own OS process
- Each worker MUST have its own CUDA context
- Workers MUST NOT share VRAM pointers
- Worker MUST own complete VRAM lifecycle (allocate → use → free)

**Rationale**: CUDA VRAM allocations are per-process. Workers need isolated VRAM ownership.

**Testing benefit**: Enables standalone worker testing (`worker-orcd --model X --gpu 0`).

---

#### [SYS-6.3.3] Tensor Parallelism Design

**Tensor Parallelism Design** (M1+):
- A single worker process MAY use multiple GPUs via CUDA for large models
- Worker maintains one CUDA context per GPU device
- NOT multiple coordinated workers (maintains isolation principle)
- Pool-managerd tracks multi-GPU workers with device mask (e.g., GPUs 0,1,2)

**Requirements**:
- Multi-GPU workers MUST be tracked with device masks in pool-managerd
- VRAM accounting MUST be per-GPU even for tensor-parallel workers
- Worker failures MUST release all GPU allocations atomically

---

#### [SYS-6.3.4] Ready Callback Contract

Worker MUST issue ready callback after initialization:
- HTTP server MUST start before the ready callback is sent (server-first)
- Callback MUST include `worker_id`, `model_ref`, `vram_bytes`, and `uri`
- Callback endpoint is provided by pool-managerd at worker startup

**Requirements**:
- Ready callback to pool-managerd MUST include `worker_id`, `model_ref`, `vram_bytes`, and `uri`
- Callback failures MUST cause worker to exit with error code
- Pool-managerd MUST update VRAM accounting and mark worker ready atomically upon receiving callback

---

#### [SYS-6.3.5] Cancellation Handling

Worker MUST handle cancellation requests promptly:
- Upon receiving `POST /cancel`, worker MUST stop decoding
- Worker MUST free resources (VRAM, buffers)
- Worker MUST emit SSE `error` event with stable code `CANCELLED`
- Worker SHOULD return HTTP 202 to acknowledge cancellation

**Requirements**:
- Cancellation MUST be idempotent (repeated cancels for same job_id are safe)
- Worker MUST complete cancellation within deadline (default 5s)
- SSE stream MUST emit terminal `error` event with `CANCELLED` code

---

## 7. Data Flow & Interactions

### 7.1 Job Submission Flow (SYS-7.1.x)

#### [SYS-7.1.1] Job Submission Flow Steps

```
1. Client → Orchestrator
   POST /v2/tasks
   { model, prompt, max_tokens, seed }

2. Orchestrator: Admission
   - Validate model exists
   - Check context length
   - Check token budget

3. Orchestrator: Enqueue
   - Add to queue (Interactive or Batch priority)
   - Return 202 Accepted + job_id

4. Orchestrator: Schedule
   - Dequeue job (Interactive first, then Batch)
   - Query pool managers for state
   - Select best worker (least-loaded, most-vram-free, round-robin)

5. Orchestrator: Dispatch
   - POST {worker_uri}/execute
   - Establish SSE stream

6. Worker: Execute
   - Load prompt
   - Run inference
   - Stream tokens via SSE

7. Orchestrator: Relay
   - Relay SSE events to client
   - Add orchestrator metadata

8. Client: Receive
   - Consume SSE stream
   - Get tokens in real-time
```

**Requirements**:
- Admission MUST validate model presence, context length, and token budgets before enqueue
- Queueing MUST respect priority classes; rejection behavior MUST follow configured policy (e.g., 429 for reject)
- Scheduling MUST select a worker based on current pool states; stale states SHOULD be refreshed if older than the heartbeat interval
- SSE relays MUST preserve event ordering and MUST propagate correlation IDs

---

### 7.2 Worker Startup Flow (SYS-7.2.x)

#### [SYS-7.2.1] Worker Startup Flow Steps

```
1. Orchestrator decides: "Need worker for model X on GPU 0"

2. Orchestrator → Pool Manager
   POST /v2/workers/start
   { model_ref: "hf:author/repo@rev::file=models/llama-7b.Q4_K_M.gguf", gpu_id: 0 }

3. Pool Manager: Preflight Validation
   - gpu-inventory: "Does GPU 0 have 16GB free?" → Yes
   - capability-matcher: "Is model compatible?" → Yes

4. Pool Manager: Spawn Worker
   worker-orcd \
     --worker-id worker-abc \
     --model /models/llama-7b.gguf \
     --gpu-device 0 \
     --port 8001 \
     --callback-url http://pool:9200/v2/internal/workers/ready

5. Worker: Initialize
   - cuda: Enforce VRAM-only, allocate VRAM
   - model-lifecycle: Load model to VRAM
   - inference-api: Start HTTP server
   - health-monitor: Start self-monitoring

6. Worker → Pool Manager (callback)
   POST /v2/internal/workers/ready
   {
     worker_id: "worker-abc",
     model_ref: "llama-7b",
     vram_bytes: 17000000000,
     uri: "http://localhost:8001"
   }

7. Pool Manager: Update State
   - gpu-inventory: Update allocated_vram
   - worker-lifecycle: Mark worker as ready

8. Pool Manager → Orchestrator (heartbeat)
   POST /v2/pools/{id}/heartbeat
   { gpus: [...], workers: [...] }

9. Orchestrator: Worker Available
   - pool-registry: Update pool state
   - scheduling: Worker now available for jobs
```

**Requirements**:
- Preflight in pool-managerd MUST validate GPU free VRAM and model compatibility before spawning the worker
- Worker-orcd MUST start HTTP server before issuing ready callback (server-first), and MUST include `model_ref` and `vram_bytes`
- Pool-managerd MUST update VRAM accounting on ready and MUST mark worker ready atomically with state update
- Orchestratord SHOULD avoid scheduling until a worker is marked ready via heartbeat; optimistic dispatch MAY be used only if configured

---

### 7.3 Worker Failure Flow (SYS-7.3.x)

#### [SYS-7.3.1] Worker Failure Flow Steps

```
1. Worker crashes (process exits)

2. Pool Manager detects (process monitoring)

3. Pool Manager: Operational Cleanup (error-ops)
   - Remove from worker registry
   - Release VRAM accounting (gpu-inventory)
   - Kill zombie processes
   - Close file handles

4. Pool Manager → Orchestrator
   POST /v2/internal/workers/failed
   {
     worker_id: "worker-abc",
     exit_code: -11,
     vram_released: 17000000000
   }

5. Orchestrator: Handle Failure
   - pool-registry: Mark worker offline
   - scheduling: Decide whether to retry
     • If retry policy allows → command new worker start
     • If not → fail pending jobs
```

**Requirements**:
- Pool-managerd MUST perform operational cleanup on crash (remove worker, release VRAM accounting, kill zombies) and MUST report failure with exit code and context
- Orchestratord MUST mark the worker offline and MUST decide retry vs. fail per configured policy (see SYS-6.1.6)
- Retries MUST apply exponential backoff per SYS-6.1.6 and MUST not violate idempotency
- If the failed worker held a scheduled or running job (tracked in `jobs` table with `worker_id` assignment), orchestrator SHOULD requeue or reassign only if policy permits and inputs are available; otherwise it MUST fail the job with a stable error code
- Metrics and logs SHOULD record failure reason, retry attempt, and outcome; proof bundles MAY capture a failure timeline for reproduction

---

### 7.4 Cancellation Flow (SYS-7.4.x)

#### [SYS-7.4.1] Cancellation Flow Steps

```
1. Client → Orchestrator
   DELETE /v2/tasks/{job_id}
   (or client disconnects SSE stream)

2. Orchestrator: Detect Cancellation
   - Client explicit cancel OR stream disconnect
   - Mark job as cancelling in state store

3. Orchestrator → Worker
   POST {worker_uri}/cancel
   { job_id: "job-xyz" }

4. Worker: Handle Cancellation
   - Stop decoding immediately
   - Free VRAM buffers
   - Emit SSE error event: { type: "error", code: "CANCELLED" }
   - Return HTTP 202 Accepted

5. Orchestrator: Confirm Cancellation
   - Receive terminal SSE event from worker
   - Update job status to "cancelled" in state store
   - Close client SSE stream (if still connected)
   - Release worker for next job

6. Orchestrator: Timeout Handling
   - If worker doesn't respond within deadline (default 5s)
   - Force-close SSE stream
   - Mark job as cancelled anyway
   - Log timeout for observability
```

**Requirements**:
- Cancellation MUST be idempotent (repeated cancels for same job_id are safe)
- Orchestrator MUST issue cancel immediately on client request or stream disconnect
- Worker MUST complete cancellation within deadline (default 5s) per SYS-6.3.5
- Orchestrator SHOULD enforce cancellation deadline; after timeout it MUST treat job as cancelled and close client SSE
- Cancellation MUST use stable error code `CANCELLED` in SSE error event
- Job state MUST transition to "cancelled" in persistent store for audit trail

---

### 7.5 SSE Reconnection Flow (SYS-7.5.x)

#### [SYS-7.5.1] SSE Reconnection Flow Steps

```
1. Client: Connection Lost
   - Network interruption or timeout
   - Client retains job_id and last received event

2. Client → Orchestrator (Reconnect)
   GET /v2/tasks/{job_id}/events
   Headers:
     Last-Event-ID: 42  (optional, last received event sequence)

3. Orchestrator: Lookup Checkpoint
   - Query sse_checkpoints table for job_id
   - Retrieve last_event_seq and last_event_type

4. Orchestrator: Resume Decision
   Case A: Job still active (running/queued)
     - Resume streaming from checkpoint
     - Replay missed events if buffered
     - Continue with live stream

   Case B: Job completed
     - Retrieve terminal event from jobs table
     - Send cached terminal event (end/error)
     - Close stream

   Case C: Job not found
     - Return 404 Not Found
     - Client should not retry

5. Client: Resume Consumption
   - Receive events from checkpoint onward
   - Update Last-Event-ID for future reconnects
```

**Requirements**:
- SSE endpoint MUST be idempotent to reconnects for the same `job_id` per SYS-5.1.3
- Orchestrator SHOULD resume streaming from the last sent offset using `sse_checkpoints` table (see SYS-6.1.3)
- Clients MAY provide `Last-Event-ID` header to indicate last received event
- If job completed during disconnect, orchestrator MUST send cached terminal event from `jobs` table
- Reconnection MUST preserve event ordering and MUST NOT duplicate events
- Checkpoint updates MUST be atomic with event emission to prevent data loss

---

## 8. Quality Attributes

### 8.1 Determinism (SYS-8.1.x)

Content already defined in Section 2.3 (Foundational Concepts). Cross-reference: See SYS-2.3.1 through SYS-2.3.5 for complete determinism requirements.

**Summary**:
- System-level guarantees: VRAM-only, pinned engines, stable event ordering
- Model-level limitations: Engines and hardware may introduce non-determinism
- Best-effort approach with recording for reproducibility

---

### 8.2 Performance (SYS-8.2.x)

#### [SYS-8.2.1] Latency Targets

**Latency targets (measurement points)**:
- Queue admission SHOULD complete within 10ms measured from HTTP receive to enqueue decision
- Scheduling decision SHOULD complete within 50ms measured from job-ready to dispatch command issued
- Worker startup SHOULD complete within 60s measured from start command to ready callback receipt (note: includes model loading time which varies by model size)
- First token latency SHOULD be under 100ms measured from worker execute accept to first SSE `token` event
- Token generation rate SHOULD be within 20–100 tokens/sec depending on model; deviations MAY be acceptable with justification in metrics

**Requirements**:
- Latency targets are guidelines, not hard requirements
- Measurements MUST be instrumented via metrics (see SYS-10.1.x)
- Deviations SHOULD be logged and investigated

---

#### [SYS-8.2.2] Throughput and Limits

**Throughput and limits**:
- Queue capacity MUST be configurable; default SHOULD be 100 jobs
- Worker concurrency for M0 MUST be 1 job per worker (batch=1)
- Multi-job batching MAY be enabled in M1+; when enabled, batching policies MUST be documented and observable via metrics

**Requirements**:
- Queue capacity of -1 indicates unbounded queue (infinite)
- Bounded queues MUST reject or evict when full per configured policy
- Throughput metrics MUST be emitted per component

---

### 8.3 Reliability (SYS-8.3.x)

#### [SYS-8.3.1] Availability

**Availability**:
- The service SHOULD achieve 99.9% uptime (3 nines) for orchestrator in supported deployments
- On worker failures, the system SHOULD degrade gracefully by failing affected jobs or rerouting per policy without impacting unrelated jobs

**Requirements**:
- Availability targets apply to orchestrator component
- Worker failures MUST NOT cascade to orchestrator
- Pool manager failures MUST be isolated per pool

---

#### [SYS-8.3.2] Fault Tolerance

**Fault tolerance**:
- Pool-managerd MUST detect worker process exits and perform operational cleanup (state removal, VRAM accounting release) promptly
- Orchestratord MUST detect missed heartbeats from pools within a configured timeout and mark them unavailable for scheduling
- Retry policies MUST be configurable with exponential backoff parameters (see SYS-6.1.6)

**Requirements**:
- Worker failures MUST be detected within 5 seconds
- Pool failures MUST be detected within 3 missed heartbeats (default 45s)
- Cleanup MUST complete within 5 seconds of detection

---

#### [SYS-8.3.3] Retry Policy

Retry policy is defined in SYS-6.1.6. Cross-reference for complete requirements.

**Summary**:
- Exponential backoff with jitter
- Configurable max attempts (default 5)
- Per-job backoff to avoid cascading failures
- Acknowledgement: Cross-worker retries MAY produce different results due to model-level non-determinism

---

#### [SYS-8.3.4] Observability Requirements

**Observability**:
- Components MUST emit structured JSON logs and Prometheus metrics sufficient to diagnose failures and capacity issues
- Key reliability events (crash, retry, timeout, cancel) SHOULD have stable event codes in logs
- Test artifacts and incident analyses MAY include proof bundles to capture timelines and seeds for reproduction

**Requirements**:
- Logs MUST include correlation IDs for tracing
- Metrics MUST conform to SYS-10.1.x
- Proof bundles SHOULD be generated per SYS-10.4.x

---

### 8.4 Scalability (SYS-8.4.x)

#### [SYS-8.4.1] Horizontal Scaling

**Horizontal scaling**:
- Orchestratord MUST support registration of multiple pool-managerd instances to scale across GPU nodes
- Pool-managerd MUST support multiple workers per node to scale with additional GPUs
- In platform mode, a platform orchestrator MAY federate across multiple provider orchestrators; routing policies SHOULD avoid nested placement decisions

**Requirements**:
- Pool registration MUST be dynamic (no restart required)
- Worker scaling MUST be per-GPU
- Federation MUST NOT nest scheduling decisions

---

#### [SYS-8.4.2] Scaling Properties

**Scaling properties**:
- Orchestrator's scheduling and registry operations SHOULD remain O(log N) or better with respect to total workers
- Metrics cardinality SHOULD be bounded when scaling; high-cardinality labels MUST be avoided or sampled

**Requirements**:
- Scheduling algorithms MUST be efficient at scale
- Metrics labels MUST NOT include unbounded identifiers (e.g., job_id)

---

#### [SYS-8.4.3] Configuration

**Configuration**:
- Scaling parameters (max pools, max workers per pool, heartbeat fan-in) SHOULD be configurable and documented

**Limits (M0)**:
- Single orchestrator instance
- Multiple pool managers supported
- Workers scale per GPU

**Limits (M1+)**:
- Orchestrator HA/clustering: future (requires PostgreSQL for shared state)
- Current: single orchestrator with SQLite persistent state (can restart gracefully)

---

## 9. Security & Compliance

### 9.1 Authentication (SYS-9.1.x)

#### [SYS-9.1.1] Authentication by Deployment Mode

Authentication requirements vary by deployment mode following the principle: **Performance > Security (Home)**, **Balanced (Lab)**, **Security > Performance (Platform)**.

**Home Mode (M0)**:
- All components on same system (localhost)
- Authentication MUST be disabled by default
- Pool registration, heartbeats, and worker callbacks are trusted (localhost)
- No bearer tokens required
- Loopback binding enforced (127.0.0.1)

**Lab Mode (M1)**:
- Orchestrator and pool-managerd on different systems
- Authentication MUST be enabled by default for network communication
- Pool registration and heartbeats MUST use bearer tokens or mTLS
- Worker callbacks within same node MAY remain unauthenticated (localhost)
- TLS SHOULD be used or run on trusted network segments

**Platform Mode (M2+)**:
- Multi-tenant marketplace, security is mandatory
- All client requests MUST be authenticated using HTTP bearer tokens
- Authentication CANNOT be disabled (security > performance)
- Pool-managerd registration MUST use bearer tokens or mTLS
- Orchestratord MUST validate tokens on every request (401/403 on failure)
- Inter-service calls SHOULD use mTLS for mutual authentication
- Audit logging MUST be enabled for all authenticated requests

**Future provisions**:
- OAuth2/OpenID Connect MAY be added in future milestones (NOT REQUIRED for M0)
- API key authentication MAY be supported as alternative to bearer tokens
- If OAuth2/OIDC is configured, orchestratord SHOULD validate audience/scope claims and enforce token expiry

---

### 9.2 EU Compliance (GDPR) (SYS-9.2.x)

#### [SYS-9.2.1] Data Residency

**Data residency**:
- MUST: Pool-managerd instances serving production workloads MUST be located within EU regions only
- MUST: Orchestratord MUST refuse to schedule work to pools that are not explicitly marked as EU-resident
- MUST: Worker-orcd MUST NOT transmit model inputs/outputs to endpoints outside the EU

---

#### [SYS-9.2.2] Geo-Verification

**Geo-verification**:
- SHOULD: Provider registration MUST include asserted region metadata and evidence (e.g., provider-declared ISO 3166-1 country code and hosting provider region ID)
- SHOULD: Orchestratord SHOULD verify region metadata against an allow-list and SHOULD reject pools with ambiguous or missing region information

---

#### [SYS-9.2.3] Data Handling

**Data handling**:
- MUST: Logs MUST NOT include raw prompts, tokens, or PII unless explicitly configured for debugging; when enabled, such logs MUST be redacted or truncated per policy
- MAY: Proof bundles MAY include hashed or redacted payloads but MUST include correlation identifiers sufficient for auditing

---

#### [SYS-9.2.4] Audit Trail

**Audit trail**:
- MUST: All requests MUST be logged with correlation IDs propagated end-to-end
- SHOULD: Access to audit logs SHOULD be role-restricted and retention policies SHOULD comply with GDPR data minimization

**Compliance docs**: `.docs/.business/monetization.md`

---

### 9.3 Multi-Tenancy (Platform Mode) (SYS-9.3.x)

#### [SYS-9.3.1] Isolation Guarantees

**Isolation guarantees**:
- Orchestratord MUST enforce tenant-level isolation of compute and data paths
- Pool-managerd MUST ensure workers for different tenants do not share processes or VRAM allocations
- Logs and metrics MUST NOT expose cross-tenant identifiers except in aggregated, non-identifying form

---

#### [SYS-9.3.2] Quotas and Limits

**Quotas and limits**:
- Orchestratord MUST support per-tenant quotas (e.g., VRAM ceilings, max concurrent jobs, token budgets)
- When quotas are exceeded, work MUST be queued (never rejected) with quota enforcement at scheduling time
- Jobs exceeding quota remain in queue until quota becomes available (tenant usage decreases)
- Submission rate limiting SHOULD be enforced per tenant on task endpoints to prevent queue flooding
- Queue capacity is unbounded (infinite queue) to prevent job loss

**Token Budget Definition** (from Glossary):
- Per-tenant quota for maximum tokens that can be generated daily or per request
- Enforced at admission time to prevent quota exhaustion
- Tracked in `tenant_quotas` table (see SYS-6.1.3)

---

#### [SYS-9.3.3] Authorization

**Authorization**:
- All client API requests in platform mode MUST include a tenant identifier bound to the authenticated principal
- Orchestratord MUST authorize every request against the tenant's entitlements before admission
- Inter-service requests (orchestrator ↔ pool-managerd) SHOULD carry tenant context for auditing; pool-managerd MAY validate tenant context when relevant

---

#### [SYS-9.3.4] Data Separation

**Data separation**:
- Shared model caches MAY be used across tenants; tenant runtime data (inputs/outputs/proofs) MUST be segregated with per-tenant namespaces
- Proof bundles MUST include correlation and tenant identifiers and SHOULD avoid raw prompts unless debugging with redaction is explicitly enabled

---

#### [SYS-9.3.5] Observability and Billing

**Observability and billing** (Platform Mode only):
- **Prometheus metrics**: MAY omit or hash `tenant_id` to reduce cardinality and privacy risk
- **Billing/accounting logs**: MUST include plaintext `tenant_id` for usage tracking
- Usage accounting MUST be recorded per tenant for tokens generated, inference duration, and VRAM occupancy-time
- Billing data MUST be stored separately from observability metrics with appropriate access controls
- Home/Lab modes: Tenancy overhead is disabled (single-user assumption)

**Specs**: `bin/orchestratord-crates/platform-api/.specs/00_platform_api.md`

---

## 10. Metrics & Observability

### 10.1 Metrics Contract (SYS-10.1.x)

#### [SYS-10.1.1] Metrics Requirements

**Orchestrator metrics**:
- `orchd_queue_depth{priority}`
- `orchd_tasks_enqueued_total{outcome}`
- `orchd_tasks_dispatched_total{worker_id, outcome}`
- `orchd_scheduling_latency_ms`

**Pool metrics**:
- `pool_mgr_gpu_vram_total_bytes{gpu_id}`
- `pool_mgr_gpu_vram_allocated_bytes{gpu_id}`
- `pool_mgr_workers_total{status}`
- `pool_mgr_worker_starts_total{outcome}`

**Worker metrics**:
- `worker_inference_duration_ms`
- `worker_tokens_generated_total`
- `worker_vram_bytes{worker_id}`

**Requirements**:
- Metric names, types, and labels MUST conform to `bin/.specs/71_metrics_contract.md`
- Required metrics MUST be emitted at INFO-level operation; optional metrics MAY be omitted but SHOULD have waivers referenced in CI (Stage 5)
- Label cardinality MUST be bounded; high-cardinality labels (e.g., `job_id`) MUST NOT be used. `worker_id` MAY be included where documented
- All metrics MUST include `component` and SHOULD include `pool_id`/`worker_id` where applicable
- Metric units MUST be encoded in the name (e.g., `_ms`, `_bytes`, `_total`)

**Spec**: `bin/.specs/71_metrics_contract.md`

---

### 10.2 Logging Standards (SYS-10.2.x)

#### [SYS-10.2.1] Log Format

**Format**:
- All components MUST emit JSON structured logs
- Log schemas SHOULD be stable and versioned to avoid breaking ingestion

**Levels**:
- Components MUST support standard levels: ERROR, WARN, INFO, DEBUG, TRACE
- Default level SHOULD be INFO in production; DEBUG/TRACE MAY be enabled temporarily for diagnostics

---

#### [SYS-10.2.2] Log Content

**Content**:
- Logs MUST include component, timestamp, level, correlation_id, and stable event codes for key actions (admission, schedule, dispatch, execute, cancel)
- Logs SHOULD avoid raw prompts/tokens unless explicitly enabled; when enabled, content MUST be redacted or truncated per policy

**Narration**:
- Human-readable narration fields MAY be included for developer ergonomics but MUST NOT replace structured fields

---

### 10.3 Correlation & Tracing (SYS-10.3.x)

#### [SYS-10.3.1] Correlation ID Propagation

Content already defined in Section 5.6. Cross-reference: See SYS-5.6.1 for complete correlation ID requirements.

**Summary**:
- `X-Correlation-Id` MUST be accepted from clients and propagated across all service calls
- If absent, orchestrator MUST generate a new correlation ID
- All logs and error responses MUST include the correlation ID
- SSE events SHOULD include correlation ID in metadata

---

### 10.4 Proof Bundle Requirements (SYS-10.4.x)

#### [SYS-10.4.1] Proof Bundle Standard

**Proof bundle standard**:
- All automated test runs SHOULD produce proof bundles under `<crate>/.proof_bundle/<type>/<run_id>/`
- Proof bundles MUST include an autogenerated header and MUST respect `LLORCH_RUN_ID` and `LLORCH_PROOF_DIR` when set
- Tests that rely on randomness MUST seed RNGs explicitly and record seeds in proof bundles

**References**:
- Proof bundle standard: `libs/proof-bundle` crate
- Spec: `.specs/00_proof-bundle.md` (if exists in monorepo root)

**Requirements**:
- Proof bundles MUST include: seeds, metadata, timestamps, correlation IDs
- Proof bundles MAY include: transcripts, timelines, redacted payloads
- Proof bundles SHOULD be deterministic (same inputs → same bundle structure)

---

## 11. Configuration

### 11.1 Orchestrator Config (SYS-11.1.x)

#### [SYS-11.1.1] Orchestrator Configuration Schema

**Orchestrator on separate machine from GPU pools**

```
[Orchestrator] (lab-controller:8080)
      ↓ (network)
[Pool Manager] (gpu-node-1:9200)
      ├─→ [Worker-1] GPU 0
      └─→ [Worker-2] GPU 1
```

**Use case**: Research lab, small team, distributed GPU resources

**Characteristics**:
- Orchestrator and pool-managerd on different systems
- Authentication enabled by default (network communication)
- Audit logging for inter-service calls
- Balanced security and performance
- Supports multiple pool managers

**Requirements**:
- Authentication MUST be enabled for orchestrator ↔ pool-managerd communication.
- Pool-managerd MUST authenticate registration and heartbeat requests using bearer tokens or mTLS.
- Orchestrator SHOULD bind to network interface (not loopback) with explicit configuration.
- Network communication SHOULD use TLS or run on trusted network segments.

---

### 5.3 Multi-GPU Mode (M1)

**Single node, multiple GPUs**

```
[Orchestrator] (localhost:8080)
      ↓
[Pool Manager] (localhost:9200)
      ├─→ [Worker-1] GPU 0 (localhost:8001)
      ├─→ [Worker-2] GPU 1 (localhost:8002)
      └─→ [Worker-3] GPU 2 (localhost:8003)
```

**Use case**: Single powerful machine with 2-4 GPUs

**Features**:
- Tensor parallelism (split large models across GPUs)
- Multiple models loaded simultaneously

**Requirements**:
- Orchestratord MUST remain single-instance with persistent state (see [SYS-0022]); restarts MUST reload queue state and allow SSE reconnection via checkpoints.
- Pool-managerd MUST manage workers: single-GPU workers map 1:1 to GPUs; multi-GPU workers (tensor parallel) map 1:N and are tracked with device masks.
- VRAM accounting MUST handle per-GPU allocation and reserved headroom; preflight MUST fail when insufficient free VRAM across required GPUs.
- Cancellation and timeouts MUST apply per-job; multi-GPU workers MUST release all GPU allocations on job termination.

---

### 5.3 Multi-Node Mode (M2+)

**Multiple nodes, multiple GPUs each**

```
[Orchestrator] (orchestrator.local:8080)
      ├─→ [Pool Manager 1] (node1:9200)
      │        ├─→ [Worker-1] GPU 0
      │        └─→ [Worker-2] GPU 1
      ├─→ [Pool Manager 2] (node2:9200)
      │        ├─→ [Worker-3] GPU 0
      │        └─→ [Worker-4] GPU 1
      └─→ [Pool Manager 3] (node3:9200)
               ├─→ [Worker-5] GPU 0
               └─→ [Worker-6] GPU 1
```

**Use case**: Enterprise, data center, GPU cluster

**Features**:
- Cluster-wide orchestration
- Load balancing across nodes
- High availability

**Requirements**:
- Orchestratord MUST manage multiple pool-managerd across nodes and MUST make all placement decisions centrally (no nested orchestrators).
- Heartbeat aggregation MUST tolerate partial failures; pools missing heartbeats beyond timeout MUST be excluded from scheduling.
- Network communication between nodes SHOULD use TLS or a trusted network; service discovery MAY be static or registry-based.
- Scheduling SHOULD prefer locality and capacity signals (e.g., free VRAM); policy MUST be observable via metrics and logs.

---

### 5.4 Platform Mode (Marketplace)

**Federation across provider orchestrators**

```
[Platform Orchestrator] (api.yourplatform.com:443)
      ├─→ [Provider A Orchestrator] (provider-a.internal:8080)
      │        └─→ Provider A's pools/workers
      ├─→ [Provider B Orchestrator] (provider-b.internal:8080)
      │        └─→ Provider B's pools/workers
      └─→ [Provider C Orchestrator] (provider-c.internal:8080)
               └─→ Provider C's pools/workers
```

**Use case**: GPU marketplace, provider ecosystem

**Features**:
- Provider registration
- Federated routing (NOT nesting)
- Billing and usage tracking
- Multi-tenancy and quotas

<!-- POTENTIAL CONTRADICTION: "smart router" vs "Provider orchestrators make their own placement decisions".
  If the platform orchestrator is "smart", what intelligence does it have? The spec should clarify
  what routing decisions it makes (e.g., provider selection based on capacity/cost/region) vs.
  what decisions it delegates (worker selection within a provider). -->
**Key distinction**: Platform orchestrator is a **smart router**, not a nested orchestrator. Provider orchestrators make their own placement decisions.

**Business doc**: `.docs/.business/monetization.md`

**Requirements**:
- Platform orchestrator MUST authenticate providers and MUST maintain a registry of provider endpoints and capabilities.
- Routing MUST be federated (provider orchestrators make placement decisions); nested scheduling MUST NOT occur.
- Tenancy context MUST be preserved across federation and included in billing records.
- Provider-level SLIs/labels (availability, capacity) SHOULD inform routing; fallbacks MAY be used when signals are missing.

---

## 6. API Contracts

### 6.0 Model Reference Format (Canonical)

**Accepted forms**:
- Client input (`model` in Agentic API) MAY be one of:
  - `hf:{org}/{repo}` or `hf:{org}/{repo}@{rev}` or `hf:{org}/{repo}@{rev}::file={path}`
  - `file:/abs/path/to/model.gguf`
  - Alias without a scheme (e.g., `llama-7b`)

<!-- MINOR INCONSISTENCY: "SHOULD pin @rev" and "SHOULD pin a concrete artifact" are both SHOULD,
  but determinism (SYS-0003) is a MUST. For true determinism guarantee, these should be MUST.
  Either strengthen these to MUST or clarify that determinism only applies when these are pinned. -->
**Resolution rules**:
- Orchestratord (catalog) MUST resolve aliases to a canonical `model_ref` prior to scheduling.
- Orchestratord SHOULD pin `@rev` to an immutable commit SHA and SHOULD pin a concrete artifact via `::file=...` for determinism.
- Pool-managerd MUST receive a normalized `model_ref` that starts with `hf:` or `file:` and MUST NOT perform alias resolution.
- The model-provisioner in pool-managerd MUST support:
  - `hf:` — Downloading the specified artifact from Hugging Face.
  - `file:` — Treating the path as a local file.
- Other schemes (e.g., `https:`, `s3:`) MAY be added in future milestones and are NOT REQUIRED for M0.

**Worker contract**:
- Worker-orcd MUST be provided a concrete filesystem path to the model artifact at startup.
- Worker ready callbacks MUST include the resolved `model_ref` for traceability.

### 6.1 Client → Orchestrator (Agentic API)

**Task submission**:
```
POST /v2/tasks
{
  "session_id": "sess-abc",
  "model": "llama-3.1-8b",
  "prompt": "Hello world",
  "max_tokens": 100,
  "temperature": 0.7,
  "seed": 42,
  "priority": "interactive"
}

Response (202 Accepted):
{
  "job_id": "job-xyz",
  "status": "queued",
  "queue_position": 2,
  "events_url": "/v2/tasks/job-xyz/events"
}
```

**Streaming**:
```
GET /v2/tasks/{job_id}/events (SSE)

Events:
- queued → started → token* → metrics* → end
- error (if failure or cancellation)
```

**Requirements**:
<!-- MINOR INCONSISTENCY: Line 307 shows example with `temperature: 0.7` but doesn't list it as required.
  Line 698 says only `model`, `prompt`, `max_tokens` are MUST.
  Should clarify if `temperature` is:
  - Optional with a default (e.g., 0.7 or 1.0)
  - Required for non-deterministic sampling
  - Not mentioned because it's part of a larger params object
-->
- Requests MUST include `model`, `prompt`, and `max_tokens`; `seed` SHOULD be provided for determinism, otherwise orchestrator MAY supply one and MUST record it.
- `priority` MUST be one of `interactive` or `batch`; unknown values MUST be rejected with 400.
<!-- MINOR INCONSISTENCY: "token budgets" is mentioned here and in line 312, but not defined anywhere.
  Is this:
  - max_tokens parameter validation (must be > 0, < some limit)?
  - Per-tenant token quotas (see line 954)?
  - Context window validation (prompt + max_tokens < model context length)?
  
  RESOLUTION NEEDED: Define "token budgets" in a glossary or earlier section.
-->
- Orchestratord MUST validate model existence and token budgets at admission; invalid requests MUST return 4xx with a stable error code.
- When the queue is full and policy=`reject`, orchestrator MUST return 429 with a Retry-After where possible; for `drop-lru`, orchestrator MUST emit metrics indicating eviction.
- SSE endpoint MUST be idempotent to reconnects for the same `job_id` and SHOULD resume streaming from the last sent offset using `sse_checkpoints` table (see [SYS-0022]); clients MAY provide `Last-Event-ID` header to indicate last received event.

**Specs**: `bin/orchestratord-crates/agentic-api/.specs/00_agentic_api.md`

---

### 6.2 Orchestrator ↔ Pool Manager

**Pool registration**:
```
POST /v2/pools/register
{
  "pool_id": "pool-1",
  "endpoint": "http://192.168.1.100:9200",
  "gpus": [...]
}
```

**Heartbeat**:
```
POST /v2/pools/{id}/heartbeat
{
  "pool_id": "pool-1",
  "gpus": [...],
  "workers": [...]
}
```

**Requirements**:
- Pool registration MUST be authenticated (see §8.1) and MUST include `pool_id`, `endpoint`, and initial GPU inventory.
- Heartbeats MUST be sent at the configured interval and MUST include current GPU VRAM state and worker states; missing heartbeats beyond timeout MUST mark the pool unavailable.
- Orchestratord MUST validate `pool_id` uniqueness and SHOULD reject re-registration attempts that conflict without explicit takeover semantics.
- Worker start commands MUST be accepted only after preflight checks (VRAM availability and compatibility) have passed in pool-managerd.

**Worker start command**:
```
POST /v2/workers/start
{
  "model_ref": "hf:author/repo@rev::file=models/model.Q4_K_M.gguf",
  "gpu_id": 0
}
```

**Specs**: `bin/pool-managerd-crates/control-api/.specs/00_control_api.md`
  (lifecycle commands and state), and `bin/pool-managerd/.specs/00_pool-managerd.md` §13 (multi-pool registration & heartbeat)

---

### 6.3 Pool Manager ↔ Worker

**Worker ready callback**:
```
POST /v2/internal/workers/ready
{
  "worker_id": "worker-abc",
  "model_ref": "llama-7b",
  "vram_bytes": 17000000000,
  "uri": "http://localhost:8001"
}
```

**Specs**: `bin/pool-managerd-crates/worker-lifecycle/.specs/00_worker_lifecycle.md`

---

### 6.4 Orchestrator → Worker (Direct)

**Inference execution**:
```
POST {worker_uri}/execute
{
  "job_id": "job-xyz",
  "prompt": "Hello world",
  "max_tokens": 100,
  "seed": 42
}

Response: SSE stream
- started
- token* (multiple)
- metrics* (periodic)
- end
- error (on failure or cancellation)
```

**Cancellation**:
```
POST {worker_uri}/cancel
{
  "job_id": "job-xyz"
}
```

Cancellation semantics:
- Idempotent: repeated cancels for the same `job_id` MUST be safe and return the same terminal outcome.
- Prompt propagation: orchestrator MUST issue cancel immediately on client request or stream disconnect.
- Worker behavior: upon cancel, worker MUST stop decoding promptly, free resources, and emit SSE `error` with a stable code `CANCELLED`.
- Acknowledgement: worker SHOULD return HTTP 202 for `POST /cancel` if cancellation has been accepted.
- Deadline: orchestrator SHOULD enforce a cancellation deadline (default 5s) after which it treats the job as cancelled and closes client SSE.

**Requirements**:
- SSE stream event order MUST be: `started` → zero or more `token` → zero or more `metrics` (interleaved as needed) → terminal `end` or `error`.
- Exactly one terminal event MUST be emitted per job (`end` on success, `error` on failure/cancel).
- Workers MUST include stable error codes; cancellation MUST use `CANCELLED`.
- Orchestrator MUST propagate `X-Correlation-Id` to worker requests; workers SHOULD echo it in SSE metadata.
- Orchestrator SHOULD enforce request timeouts per job policy and close SSE cleanly on timeout with a terminal event.

**Specs**: `bin/worker-orcd/.specs/00_worker-orcd.md`

---

## 7. Quality Attributes

### 7.1 Determinism

**Design Principle**: The system is designed AS IF deterministic, but acknowledges that underlying models may not guarantee determinism.

**System-level guarantees** (what we control):
- Worker-orcd MUST allocate and keep all model weights, KV cache, and activations in VRAM only
- Engine versions and kernel parameters MUST be pinned and recorded per job
- SSE event ordering MUST be stable and reproducible
- Same seed + same inputs MUST follow identical code paths through the system

**Model-level limitations** (what we cannot control):
- Inference engines (llama.cpp, vLLM, etc.) MAY have non-deterministic operations
- GPU hardware variations MAY produce different floating-point results
- Model architectures MAY include inherently non-deterministic components
- Cross-worker/cross-GPU determinism is NOT guaranteed

**Best-effort determinism**:
- When model and engine support determinism, system MUST preserve it
- Sampling SHOULD be deterministic for identical inputs and seeds where engine allows
- Non-deterministic operations SHOULD be disabled or replaced when possible
- System MUST document which models/engines have been verified as deterministic

**Recording for reproducibility**:
- Proof bundles MUST include seed, model_ref (pinned @rev and artifact), engine version, device info
- Failed determinism attempts SHOULD be logged with hardware/engine context

**Verification approach**:
- Property tests SHOULD verify determinism for known-deterministic models
- Research document MUST catalog which models/engines achieve determinism
- See `.docs/determinism-research.md` for model-specific findings

---

### 7.2 Performance

<!-- MINOR INCONSISTENCY: "Worker startup SHOULD complete within 60s" but this is a pool-managerd responsibility,
  not directly controllable by the spec. The 60s includes model loading time which varies by model size.
  Consider whether this should be a SHOULD or just a target/guideline. -->
**Latency targets (measurement points)**:
- Queue admission SHOULD complete within 10ms measured from HTTP receive to enqueue decision.
- Scheduling decision SHOULD complete within 50ms measured from job-ready to dispatch command issued.
- Worker startup SHOULD complete within 60s measured from start command to ready callback receipt.
- First token latency SHOULD be under 100ms measured from worker execute accept to first SSE `token` event.
- Token generation rate SHOULD be within 20–100 tokens/sec depending on model; deviations MAY be acceptable with justification in metrics.

**Throughput and limits**:
- Queue capacity MUST be configurable; default SHOULD be 100 jobs.
- Worker concurrency for M0 MUST be 1 job per worker (batch=1).
- Multi-job batching MAY be enabled in M1+; when enabled, batching policies MUST be documented and observable via metrics.

---

### 7.3 Reliability

**Availability**:
- The service SHOULD achieve 99.9% uptime (3 nines) for orchestrator in supported deployments.
- On worker failures, the system SHOULD degrade gracefully by failing affected jobs or rerouting per policy without impacting unrelated jobs.
- Automatic retry with exponential backoff SHOULD be applied where policies permit.
- Retry behavior acknowledges that cross-worker retries MAY produce different results (see §7.1 Determinism).
- Retries are for fault tolerance, not determinism guarantees.

**Fault tolerance**:
- Pool-managerd MUST detect worker process exits and perform operational cleanup (state removal, VRAM accounting release) promptly.
- Orchestratord MUST detect missed heartbeats from pools within a configured timeout and mark them unavailable for scheduling.
- Retry policies MUST be configurable with exponential backoff parameters.

**Retry backoff configuration**:
```yaml
retry:
  enabled: true
  max_attempts: 5
  initial_delay_ms: 1000  # 1 second
  multiplier: 2.0  # exponential: 1s, 2s, 4s, 8s, 16s
  max_delay_ms: 60000  # cap at 60 seconds
  jitter: true  # add random jitter to prevent thundering herd
```

**Observability**:
- Components MUST emit structured JSON logs and Prometheus metrics sufficient to diagnose failures and capacity issues.
- Key reliability events (crash, retry, timeout, cancel) SHOULD have stable event codes in logs.
- Test artifacts and incident analyses MAY include proof bundles to capture timelines and seeds for reproduction.

---

### 7.4 Scalability

**Horizontal scaling**:
- Orchestratord MUST support registration of multiple pool-managerd instances to scale across GPU nodes.
- Pool-managerd MUST support multiple workers per node to scale with additional GPUs.
- In platform mode, a platform orchestrator MAY federate across multiple provider orchestrators; routing policies SHOULD avoid nested placement decisions.

**Scaling properties**:
- Orchestrator’s scheduling and registry operations SHOULD remain O(log N) or better with respect to total workers.
- Metrics cardinality SHOULD be bounded when scaling; high-cardinality labels MUST be avoided or sampled.

**Configuration**:
- Scaling parameters (max pools, max workers per pool, heartbeat fan-in) SHOULD be configurable and documented.

**Limits (M0)**:
- Single orchestrator instance
- Multiple pool managers supported
- Workers scale per GPU

**Limits (M1+)**:
- Orchestrator HA/clustering: future (requires PostgreSQL for shared state)
- Current: single orchestrator with SQLite persistent state (can restart gracefully)

---

## 8. Security & Compliance

### 8.1 Authentication

Authentication requirements vary by deployment mode following the principle: **Performance > Security (Home)**, **Balanced (Lab)**, **Security > Performance (Platform)**.

**Home Mode (M0)**:
- All components on same system (localhost)
- Authentication MUST be disabled by default
- Pool registration, heartbeats, and worker callbacks are trusted (localhost)
- No bearer tokens required
- Loopback binding enforced (127.0.0.1)

**Lab Mode (M1)**:
- Orchestrator and pool-managerd on different systems
- Authentication MUST be enabled by default for network communication
- Pool registration and heartbeats MUST use bearer tokens or mTLS
- Worker callbacks within same node MAY remain unauthenticated (localhost)
- TLS SHOULD be used or run on trusted network segments

**Platform Mode (M2+)**:
- Multi-tenant marketplace, security is mandatory
- All client requests MUST be authenticated using HTTP bearer tokens
- Authentication CANNOT be disabled (security > performance)
- Pool-managerd registration MUST use bearer tokens or mTLS
- Orchestratord MUST validate tokens on every request (401/403 on failure)
- Inter-service calls SHOULD use mTLS for mutual authentication
- Audit logging MUST be enabled for all authenticated requests

**Future provisions**:
- OAuth2/OpenID Connect MAY be added in future milestones (NOT REQUIRED for M0)
- API key authentication MAY be supported as alternative to bearer tokens
- If OAuth2/OIDC is configured, orchestratord SHOULD validate audience/scope claims and enforce token expiry

---

### 8.2 EU Compliance (GDPR)

**Data residency**:
- MUST: Pool-managerd instances serving production workloads MUST be located within EU regions only.
- MUST: Orchestratord MUST refuse to schedule work to pools that are not explicitly marked as EU-resident.
- MUST: Worker-orcd MUST NOT transmit model inputs/outputs to endpoints outside the EU.

**Geo-verification**:
- SHOULD: Provider registration MUST include asserted region metadata and evidence (e.g., provider-declared ISO 3166-1 country code and hosting provider region ID).
- SHOULD: Orchestratord SHOULD verify region metadata against an allow-list and SHOULD reject pools with ambiguous or missing region information.

**Data handling**:
- MUST: Logs MUST NOT include raw prompts, tokens, or PII unless explicitly configured for debugging; when enabled, such logs MUST be redacted or truncated per policy.
- MAY: Proof bundles MAY include hashed or redacted payloads but MUST include correlation identifiers sufficient for auditing.

**Audit trail**:
- MUST: All requests MUST be logged with correlation IDs propagated end-to-end.
- SHOULD: Access to audit logs SHOULD be role-restricted and retention policies SHOULD comply with GDPR data minimization.

---
**Compliance docs**: `.docs/.business/monetization.md`

---

### 8.3 Multi-Tenancy (Platform Mode)

**Isolation guarantees**:
- Orchestratord MUST enforce tenant-level isolation of compute and data paths.
- Pool-managerd MUST ensure workers for different tenants do not share processes or VRAM allocations.
- Logs and metrics MUST NOT expose cross-tenant identifiers except in aggregated, non-identifying form.

**Quotas and limits**:
- Orchestratord MUST support per-tenant quotas (e.g., VRAM ceilings, max concurrent jobs, token budgets).
- When quotas are exceeded, work MUST be queued (never rejected) with quota enforcement at scheduling time.
- Jobs exceeding quota remain in queue until quota becomes available (tenant usage decreases).
- Submission rate limiting SHOULD be enforced per tenant on task endpoints to prevent queue flooding.
- Queue capacity is unbounded (infinite queue) to prevent job loss.

**Authorization**:
- All client API requests in platform mode MUST include a tenant identifier bound to the authenticated principal.
- Orchestratord MUST authorize every request against the tenant’s entitlements before admission.
- Inter-service requests (orchestrator ↔ pool-managerd) SHOULD carry tenant context for auditing; pool-managerd MAY validate tenant context when relevant.

**Data separation**:
- Shared model caches MAY be used across tenants; tenant runtime data (inputs/outputs/proofs) MUST be segregated with per-tenant namespaces.
- Proof bundles MUST include correlation and tenant identifiers and SHOULD avoid raw prompts unless debugging with redaction is explicitly enabled.

**Observability and billing** (Platform Mode only):
- **Prometheus metrics**: MAY omit or hash `tenant_id` to reduce cardinality and privacy risk
- **Billing/accounting logs**: MUST include plaintext `tenant_id` for usage tracking
- Usage accounting MUST be recorded per tenant for tokens generated, inference duration, and VRAM occupancy-time
- Billing data MUST be stored separately from observability metrics with appropriate access controls
- Home/Lab modes: Tenancy overhead is disabled (single-user assumption)

**Specs**: `bin/orchestratord-crates/platform-api/.specs/00_platform_api.md`

---

## 9. Development Workflow

### 9.1 Spec-Driven Development

**Process**: Spec → Contract → Tests → Code

**Workflow**:
1. Write spec (RFC-2119 normative requirements)
2. Define contracts (API, data structures)
3. Write tests (BDD, property tests, unit tests)
4. Implement code (guided by specs and tests)

**Docs**: `README_LLM.md`, `.docs/workflow.md`

---

### 9.2 Testing Strategy

**Test types (scope and requirements)**:
- Unit tests MUST exist per crate to cover core functions and error paths.
- Integration tests SHOULD validate cross-crate interactions and HTTP boundaries.
- Contract tests MUST verify API conformance against the OpenAPI/contracts with stable IDs.
- Property tests MUST enforce critical invariants (e.g., determinism, queue bounds, idempotent cancellation).
- BDD tests SHOULD derive from spec scenarios with stable identifiers and traceability to requirements.

**Artifacts and determinism**:
- All automated test runs SHOULD produce proof bundles under `<crate>/.proof_bundle/<type>/<run_id>/`.
- Proof bundles MUST include an autogenerated header and MUST respect `LLORCH_RUN_ID` and `LLORCH_PROOF_DIR` when set.
- Tests that rely on randomness MUST seed RNGs explicitly and record seeds in proof bundles.

**References**:
- Proof bundle standard: `.specs/00_proof-bundle.md`

---

### 9.3 CI/CD Pipeline

**Gates (must-pass checks)**:
- Stage 0 (Spec hygiene) MUST pass link checks and ID stability; specs MUST use RFC‑2119 terms.
- Stage 1 (Code quality) MUST pass `fmt` and `clippy` with no new warnings in changed code.
- Stage 2 (Tests) MUST pass unit, integration, and property tests; flaky tests MUST be quarantined or fixed before merge.
- Stage 3 (Contract compliance) MUST validate APIs against OpenAPI/contracts; breaking changes MUST be versioned and documented.
- Stage 4 (Determinism suite) MUST verify determinism properties on supported targets.
- Stage 5 (Metrics emission) SHOULD validate presence and type/label cardinality of required metrics; missing optional metrics MAY be tolerated with waivers.
- Stage 6 (E2E acceptance) SHOULD pass scenario-based BDD tests for targeted milestones before release.

**Planning references**:
- Roadmap documents SHOULD be maintained at `TODO.md` and `.plan/00_meta_plan.md`.

---

## 10. Metrics & Observability

### 10.1 Metrics Contract

**Orchestrator metrics**:
- `orchd_queue_depth{priority}`
- `orchd_tasks_enqueued_total{outcome}`
- `orchd_tasks_dispatched_total{worker_id, outcome}`
- `orchd_scheduling_latency_ms`

**Pool metrics**:
- `pool_mgr_gpu_vram_total_bytes{gpu_id}`
- `pool_mgr_gpu_vram_allocated_bytes{gpu_id}`
- `pool_mgr_workers_total{status}`
- `pool_mgr_worker_starts_total{outcome}`

**Worker metrics**:
- `worker_inference_duration_ms`
- `worker_tokens_generated_total`
- `worker_vram_bytes{worker_id}`

**Requirements**:
- Metric names, types, and labels MUST conform to `bin/.specs/71_metrics_contract.md`.
- Required metrics MUST be emitted at INFO-level operation; optional metrics MAY be omitted but SHOULD have waivers referenced in CI (Stage 5).
- Label cardinality MUST be bounded; high-cardinality labels (e.g., `job_id`) MUST NOT be used. `worker_id` MAY be included where documented.
- All metrics MUST include `component` and SHOULD include `pool_id`/`worker_id` where applicable.
- Metric units MUST be encoded in the name (e.g., `_ms`, `_bytes`, `_total`).

**Spec**: `.specs/71_metrics_contract.md`

---

### 10.2 Logging

**Format**:
- All components MUST emit JSON structured logs.
- Log schemas SHOULD be stable and versioned to avoid breaking ingestion.

**Levels**:
- Components MUST support standard levels: ERROR, WARN, INFO, DEBUG, TRACE.
- Default level SHOULD be INFO in production; DEBUG/TRACE MAY be enabled temporarily for diagnostics.

**Correlation**:
- `X-Correlation-Id` MUST be accepted from clients and propagated across orchestrator → pool-managerd → worker calls.
- If the header is absent, orchestrator MUST generate a new correlation ID and propagate it downstream.

**Content**:
- Logs MUST include component, timestamp, level, correlation_id, and stable event codes for key actions (admission, schedule, dispatch, execute, cancel).
- Logs SHOULD avoid raw prompts/tokens unless explicitly enabled; when enabled, content MUST be redacted or truncated per policy.

**Narration**:
- Human-readable narration fields MAY be included for developer ergonomics but MUST NOT replace structured fields.

---

## 11. Configuration

### 11.1 Orchestrator Config

```yaml
orchestratord:
  bind: "0.0.0.0:8080"
  mode: "agentic"  # or "platform"
  
  queue:
    capacity: -1  # -1 = unbounded (infinite queue), or positive integer for bounded
    policy: "queue"  # always queue, never reject
  
  scheduling:
    algorithm: "least-loaded"  # or "most-vram-free", "round-robin"
    eviction_policy: "lru"     # or "lfu", "manual"
    
  eviction:
    # Two eviction scenarios:
    # 1. Model eviction: Remove cached models from RAM when no longer needed
    # 2. Worker eviction: Stop workers to free VRAM for higher-priority jobs
    model_cache_policy: "lru"  # evict least-recently-used models from RAM cache
    worker_policy: "lru"       # stop least-recently-used workers to free VRAM
    vram_threshold: 0.9        # trigger worker eviction when VRAM > 90% utilized
  
  timeout:
    default_ms: 300000  # 5 minutes
    max_ms: 1800000     # 30 minutes
```

**Requirements**:
- Defaults MUST be applied when fields are omitted; values MUST be validated (e.g., `default_ms` ≤ `max_ms`).
- `mode` MUST be `agentic` or `platform`; unknown values MUST be rejected.
- `queue.capacity`: -1 for unbounded (default), or positive integer for bounded queue.
- `queue.policy`: MUST be `queue` (never reject jobs, always enqueue).
- `scheduling.algorithm` SHOULD be one of the documented strategies; unknown algorithms MAY be rejected or fall back to a safe default with a warning.
- Timeouts MUST be enforced per job; values MAY be overridden per-request if allowed by policy.

---

### 11.2 Pool Manager Config

```yaml
pool-managerd:
  bind: "0.0.0.0:9200"
  pool_id: "pool-1"
  
  orchestrator:
    url: "http://orchestrator:8080"
    heartbeat_interval_ms: 15000
  
  models:
    cache_dir: "/var/cache/llama-orch/models"
```

**Requirements**:
- `bind` MUST be a valid socket address; in home mode it SHOULD bind loopback unless explicitly overridden.
- `pool_id` MUST be unique per orchestrator; collisions MUST be rejected at registration.
- `orchestrator.url` MUST be a reachable HTTP endpoint; `heartbeat_interval_ms` MUST be a positive integer and SHOULD default to 15000.
- `models.cache_dir` MUST be writable by pool-managerd; insufficient permissions MUST fail fast at startup.

---

### 11.3 Worker Config

```bash
worker-orcd \
  --worker-id worker-abc \
  --model /models/llama-7b.gguf \
  --gpu-device 0 \
  --port 8001 \
  --callback-url http://pool:9200/v2/internal/workers/ready
```

**Requirements**:
- `--worker-id` MUST be unique per pool; collisions MUST be rejected.
- `--model` MUST point to a readable file; startup MUST fail fast if missing or unreadable.
- `--gpu-device` MUST refer to a valid CUDA device index present on the node.
- `--port` SHOULD default to an ephemeral port if not provided; chosen port MUST be free at startup.
- `--callback-url` MUST be reachable by pool-managerd and MUST use HTTP(S); ready callback MUST include `model_ref` and `vram_bytes`.

---

## 12. Crate Dependency Graph

```
orchestratord
├── scheduling (admission, queue, job-tracker, scheduler, eviction)
├── platform-api (marketplace facade)
├── agentic-api (standard API)
├── pool-registry (track pools)
├── streaming (SSE relay)
├── task-cancellation (cancel propagation)
├── job-timeout (timeout enforcement)
└── backpressure (backpressure handling)

pool-managerd
├── gpu-inventory (NVML FFI)
├── capability-matcher (preflight validation)
├── model-cache (storage)
├── model-provisioner (download)
├── model-catalog (metadata)
├── worker-lifecycle (spawn/monitor workers)
├── control-api (HTTP API)
├── error-ops (operational cleanup)
└── pool-registration-client (register with orchestrator)

worker-orcd
├── http (HTTP handlers: execute, health)
├── startup (initialization and callbacks)
├── cuda (FFI to C++/CUDA)
├── health-monitor (self-monitoring)
└── error-handler (error handling)
```

---

## 13. Milestone Roadmap

### M0: Single GPU (v0.1.0)

**Goal**: Home lab, single user, single GPU

**Features**:
- Single orchestrator instance
- Single pool manager
- Single worker per GPU
- VRAM-only enforcement
- Determinism guarantee
- Basic API (task submission, streaming)

**Status**: In progress

---

### M1: Multi-GPU (v0.2.0)

**Goal**: Single node, multiple GPUs

**Features**:
- Tensor parallelism (single worker process using multiple GPUs)
- Multiple workers per node
- Model hot-loading (pool-level optimization: models preloaded in RAM/page cache for fast worker replacement)

---

### M2: Multi-Node (v0.3.0)

**Goal**: Enterprise, GPU cluster

**Features**:
- Multiple pool managers
- Cluster-wide orchestration
- Load balancing across nodes

---

### M3: Platform (v0.4.0)

**Goal**: GPU marketplace

**Features**:
- Provider registration
- Federated routing
- Billing and usage tracking
- Multi-tenancy

---

## Non-goals / Out of Scope

- RAM fallback for model weights, KV cache, or activations — NOT SUPPORTED (VRAM-only policy applies).
- CUDA Unified Memory (UMA) and zero-copy modes — NOT SUPPORTED.
- Disk swapping or spill for inference state — NOT SUPPORTED.
- CPU inference fallback — NOT SUPPORTED.
- Nested schedulers/orchestrators — NOT SUPPORTED; platform routing is federated, not nested (see §5.4).

---

## 14. References

### Specifications

**Component specs**:
- `bin/orchestratord/.specs/00_orchestratord.md` (ORCH-1xxx)
- `bin/pool-managerd/.specs/00_pool-managerd.md` (POOL-2xxx)
- `bin/worker-orcd/.specs/00_worker-orcd.md` (WORK-3xxx)

**Crate specs**: See individual crate `.specs/` directories

### Documentation

- `README_LLM.md` — AI-optimized project overview
- `.docs/workflow.md` — Development workflow
- `.docs/.business/monetization.md` — Marketplace business model
- `.docs/test-case-discovery-method.md` — Testing approach
- `TODO.md` — Active roadmap
- `.plan/00_meta_plan.md` — Project plan

### Contracts

- `/contracts/openapi/*.yaml` — OpenAPI specs

---

## 15. Traceability

**Code**: Monorepo at `/home/vince/Projects/llama-orch/`  
**Tests**: Per-crate `tests/` directories, BDD features in `bdd/`  
**Artifacts**: `.proof_bundle/<type>/<run_id>/`  
**CI**: GitHub Actions (`.github/workflows/`)  
**Spec IDs**: SYS-0xxx (this document)

---

**Version**: 0.1.0  
**Last Updated**: 2025-10-03  
**Status**: Living document (updated as system evolves)

---

**End of System Specification**
