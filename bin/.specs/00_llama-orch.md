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
- [2.2 Memory Architecture Policy](#22-memory-architecture-policy-sys-22x)
- [2.3 Determinism Principles](#23-determinism-principles-sys-23x)
- [2.4 Process Isolation Rationale](#24-process-isolation-rationale-sys-24x)
- [2.5 FFI Boundaries](#25-ffi-boundaries-sys-25x)
### 3. Deployment Modes
- [3.1 Home Mode (M0)](#31-home-mode-m0-sys-31x)
- [3.2 Lab Mode (M2)](#32-lab-mode-m2-sys-32x)
- [3.3 Multi-GPU Mode (M4)](#33-multi-gpu-mode-m4-sys-33x)
- [3.4 Multi-Node Mode (M4)](#34-multi-node-mode-m4-sys-34x)
- [3.5 Platform Mode (M3+)](#35-platform-mode-m3-sys-35x)
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
- [6.1 Orchestratord (The Brain) [M2]](#61-queen-rbee-the-brain-sys-61x)
  - [6.1.1 Orchestrator Intelligence](#611-orchestrator-intelligence-sys-611)
  - [6.1.2 State Management](#612-state-management-sys-612)
  - [6.1.3 Persistent State Store](#613-persistent-state-store-sys-613)
  - [6.1.4 Queue Optimizer](#614-queue-optimizer-sys-614)
  - [6.1.5 Programmable Scheduler](#615-programmable-scheduler-sys-615)
  - [6.1.6 Retry & Backoff Policy](#616-retry--backoff-policy-sys-616)
- [6.2 Pool-Managerd (Control Plane) [M1]](#62-pool-managerd-control-plane-sys-62x)
  - [6.2.1 Pool Manager Execution](#621-pool-manager-execution-sys-621)
  - [6.2.2 State Reporting](#622-state-reporting-sys-622)
  - [6.2.3 Preflight Validation](#623-preflight-validation-sys-623)
  - [6.2.4 Heartbeat Protocol](#624-heartbeat-protocol-sys-624)
  - [6.2.5 Operational Cleanup](#625-operational-cleanup-sys-625)
- [6.3 Worker-Orcd (Executor) [M0]](#63-worker-orcd-executor-sys-63x)
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
### 9. Security & Compliance [M3]
- [9.1 Authentication](#91-authentication-sys-91x)
- [9.2 EU Compliance (GDPR)](#92-eu-compliance-gdpr-sys-92x)
- [9.3 Multi-Tenancy (Platform Mode)](#93-multi-tenancy-platform-mode-sys-93x)
### 10. Metrics & Observability [M-1+]
- [10.1 Metrics Contract](#101-metrics-contract-sys-101x)
- [10.2 Logging Standards](#102-logging-standards-sys-102x)
- [10.3 Correlation & Tracing](#103-correlation--tracing-sys-103x)
- [10.4 Proof Bundle Requirements](#104--requirements-sys-104x)
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
- [14.0 M-1: Foundation (Pre-M0)](#140-m-1-foundation-pre-m0)
- [14.1 M0: Worker Haiku Test (v0.1.0)](#141-m0-worker-haiku-test-v010)
- [14.2 M1: Pool Manager Lifecycle (v0.2.0)](#142-m1-pool-manager-lifecycle-v020)
- [14.3 M2: Orchestrator Scheduling (v0.3.0)](#143-m2-orchestrator-scheduling-v030)
- [14.4 M3: Security & Platform Readiness (v0.4.0)](#144-m3-security--platform-readiness-v040)
- [14.5 M4: Multi-GPU & Multi-Node (v0.5.0)](#145-m4-multi-gpu--multi-node-v050)
- [14.6 M5: Platform Marketplace (v0.6.0)](#146-m5-platform-marketplace-v060)
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
- **Eviction**: Two types: (1) Model eviction (hot-load eviction) - removing cached model files from pool-managerd RAM to free memory when no longer needed for quick worker startup; (2) Worker eviction - stopping worker processes to free VRAM for higher-priority jobs.
- **Preflight Validation**: Pre-spawn checks performed by pool-managerd to verify GPU has sufficient free VRAM and model is compatible before spawning a worker.
- **VRAM-Only Policy**: Worker-specific requirement for bespoke NVIDIA workers (worker-orcd) that all model weights, KV cache, activations, and intermediate tensors reside entirely in GPU VRAM with no RAM/disk fallback. Other worker types (e.g., Apple ARM workers) may use unified memory architecture as appropriate for their platform.
- **Test Reproducibility**: Property where same model + same seed + temp=0 + same prompt produces identical output for testing validation; NOT a product guarantee due to model and hardware limitations.
- **Proof Bundle**: Standardized test artifact directory containing seeds, transcripts, metadata, and outputs for reproducibility (see `libs/`).
- **Smart/Dumb Boundary**: Architectural principle where queen-rbee makes ALL intelligent decisions (smart) while pool-managerd and workers execute commands without policy decisions (dumb).
- **Model Reference (model_ref)**: Canonical identifier for a model artifact, format: `hf:{org}/{repo}@{rev}::file={path}` or `file:/path/to/model.gguf`.
- **Model Adapter**: Component within worker-orcd that abstracts architecture-specific inference logic (e.g., Llama-style vs GPT-style models). Distinct from worker adapter (pool-managerd component for worker lifecycle).
- **Worker Adapter**: Component in pool-managerd that abstracts worker lifecycle for different worker types (bespoke-cuda, llama.cpp, apple-metal). See `bin/pool-managerd/.specs/10_worker_adapters.md` (POOL-1xxx). Distinct from model adapter (worker-internal architecture abstraction).
- **Worker Type**: Category of worker implementation (bespoke-cuda, llamacpp, apple-metal, image-gen). Pool manager uses worker type to select appropriate adapter for spawning.
- **Worker Capability**: Functional capability advertised by worker (text-gen, image-gen, audio-gen, embedding, multimodal). Orchestrator routes jobs based on capability, not worker type.
- **Streaming Protocol**: Response format used by worker for a given capability (SSE for text-gen, JSON for image-gen/embedding, Binary for audio-gen, Mixed SSE for multimodal). Orchestrator selects relay strategy based on protocol.
- **Heartbeat**: Periodic state report from pool-managerd to queen-rbee (default 15s interval) containing GPU VRAM state and worker status.
- **SSE (Server-Sent Events)**: HTTP streaming protocol used for token-by-token inference results from worker → orchestrator → client.
- **Correlation ID**: Unique identifier (`X-Correlation-Id` header) propagated across all service calls for request tracing and log correlation.
- **Tenant**: Isolated customer/user in platform mode with separate quotas, billing, and resource allocation.
- **Priority Classes**: Job queue priorities - `interactive` (user-facing, low latency) and `batch` (background, high throughput).
### 0.2 Traceability Index
**Quick Lookup Table:**
| ID Range | Section | Description |
|----------|---------|-------------|
| SYS-2.1.x | Foundational Concepts | Model Reference Format |
| SYS-2.2.x | Foundational Concepts | Memory Architecture Policy |
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
| SYS-5.7.x | API Contracts | Multi-Modality Protocols |
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
Llama-Orch is a **multi-node GPU orchestration system** for large language model inference. It provides test reproducibility, EU-native compliance, and enterprise-grade orchestration across distributed GPU resources with support for multiple worker types and memory architectures.
**Core Value Propositions:**
1. **Test Reproducibility**: Same seed + temp=0 → Same output (for testing validation only)
2. **Multi-Architecture Support**: NVIDIA CUDA workers (VRAM-only), Apple ARM workers (unified memory), and extensible worker types
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
## 2. Foundational Concepts [All Milestones]
### 2.1 Model Reference Format [M0+] (SYS-2.1.x)
#### [SYS-2.1.1] Model Reference Accepted Forms
**Accepted forms**:
- Client input (`model` in Agentic API) MAY be one of:
  - `hf:{org}/{repo}` or `hf:{org}/{repo}@{rev}` or `hf:{org}/{repo}@{rev}::file={path}`
  - `file:/abs/path/to/model.gguf`
  - Alias without a scheme (e.g., `llama-7b`)
#### [SYS-2.1.2] Model Reference Resolution
**Resolution rules**:
- Orchestratord (catalog) MUST resolve aliases to a canonical `model_ref` prior to scheduling.
- Orchestratord MUST pin `@rev` to an immutable commit SHA and MUST pin a concrete artifact via `::file=...` for test reproducibility (strengthened from SHOULD to align with SYS-2.3.1).
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
### 2.2 Memory Architecture Policy [M0+] (SYS-2.2.x)
#### [SYS-2.2.1] Worker-Specific Memory Requirements
Worker implementations MUST declare their memory architecture requirements. The orchestrator and pool manager MUST NOT enforce a specific memory architecture—this is worker-specific.
**Worker types and their memory policies:**
1. **Bespoke NVIDIA Workers (worker-orcd)**: VRAM-ONLY policy (see `bin/.specs/01_M0_worker_orcd.md`)
   - Model weights, KV cache, activations, and intermediate tensors MUST reside entirely in GPU VRAM
   - Prohibited: RAM fallback, Unified memory (CUDA UMA), Zero-copy mode, CPU inference fallback, Disk swapping
   - Rationale: Ensures predictable performance and enables test reproducibility by eliminating memory hierarchy variability
2. **Apple ARM Workers (worker-aarmd)**: UNIFIED MEMORY architecture (future, see `bin/pool-managerd/.specs/10_worker_adapters.md` and `bin/worker-aarmd/.specs/00_worker-aarmd.md`)
   - MUST use Apple Metal unified memory architecture
   - Model weights and activations reside in unified memory accessible by both CPU and GPU
   - Rationale: Leverages Apple Silicon architecture for efficient memory usage
   - Binary: `bin/worker-aarmd/` (Apple ARM daemon)
3. **Other worker types**: Define memory requirements in their respective specs
**System-level requirement**: Pool manager MUST report worker memory architecture to orchestrator for scheduling decisions. Orchestrator MUST NOT make assumptions about worker memory layout.
---
### 2.3 Test Reproducibility Principles [M0+] (SYS-2.3.x)
#### [SYS-2.3.1] Test Reproducibility (NOT a Product Guarantee)
The system MUST provide reproducibility for testing: same model + same seed + temp=0 + same prompt → same output (for validation only, NOT a product guarantee).
**Requirements:**
- Sealed memory allocation (worker-specific: VRAM shards for worker-orcd, unified memory for worker-aarmd)
- Pinned engine versions
- Temperature-based sampling (M0: 0.0-2.0 supported; temp=0 for testing reproducibility)
- Batch=1 (single request at a time)
- No non-deterministic operations in system code
- Quantized execution (same quantization format → same numerical results)
**Design Principle**: The system provides REPRODUCIBILITY for testing (temp=0 + same seed → same output), but this is NOT a product promise. Models cannot guarantee deterministic behavior due to model architecture and hardware limitations. Temperature-based sampling (0.0-2.0) is the product feature.
#### [SYS-2.3.2] System-Level Guarantees
**System-level guarantees** (what we control):
- Workers MUST allocate and keep all model weights, KV cache, and activations in their designated memory architecture (VRAM-only for worker-orcd, unified memory for worker-aarmd)
- Engine versions and kernel parameters MUST be pinned and recorded per job
- SSE event ordering MUST be stable and reproducible
- Same seed + same inputs MUST follow identical code paths through the system
#### [SYS-2.3.3] Model-Level Limitations
**Model-level limitations** (what we cannot control):
- Inference engines (llama.cpp, vLLM, etc.) MAY have non-deterministic operations
- GPU hardware variations MAY produce different floating-point results
- Model architectures MAY include inherently non-deterministic components
- Cross-worker/cross-GPU determinism is NOT guaranteed
#### [SYS-2.3.4] Best-Effort Test Reproducibility
**Best-effort test reproducibility**:
- When model and engine support reproducibility, system MUST preserve it
- Sampling SHOULD be reproducible for identical inputs and seeds where engine allows
- Non-deterministic operations SHOULD be disabled or replaced when possible
- System MUST document which models/engines have been verified as reproducible for testing
#### [SYS-2.3.5] Recording for Reproducibility
**Recording for reproducibility**:
-  MUST include seed, model_ref (pinned @rev and artifact), engine version, device info
- Failed reproducibility attempts SHOULD be logged with hardware/engine context
- Property tests SHOULD verify reproducibility for known-reproducible models
- Research document MUST catalog which models/engines achieve test reproducibility (see `.docs/reproducibility-research.md`)
---
### 2.4 Process Isolation Rationale [M0+] (SYS-2.4.x)
#### [SYS-2.4.1] Process Isolation Requirement
Workers MUST run in separate processes from pool managers.
**Why**: Hardware-specific memory allocations (e.g., CUDA VRAM, Metal unified memory) are per-process. Workers need self-contained memory ownership within their execution context.
#### [SYS-2.4.2] Worker Process Isolation
Workers MUST run in separate processes:
**Requirements**:
- Each worker MUST have its own OS process
- Each worker MUST have its own hardware context (e.g., CUDA context for NVIDIA workers, Metal context for Apple ARM workers)
- Workers MUST NOT share memory pointers across processes
- Worker MUST own complete memory lifecycle (allocate → use → free)
**Rationale**: Hardware-specific memory allocations are per-process. Workers need isolated memory ownership within their execution context.
**Testing benefit**: Enables standalone worker testing (`worker-orcd --model X --gpu 0`).
**Communication**: Components MUST communicate via HTTP APIs only.
---
### 2.5 FFI Boundaries [M0+] (SYS-2.5.x)
#### [SYS-2.5.1] FFI Boundary Enforcement
The system MUST enforce strict FFI boundaries:
**Pool manager** (NVML only):
- MUST use NVML for read-only GPU queries
- MUST query system-wide state (all GPUs)
- MUST NOT allocate VRAM or use CUDA
- MUST NOT perform compute operations
**Worker** (architecture-specific):
- Bespoke NVIDIA workers (worker-orcd): MUST use CUDA Runtime API for VRAM allocation within process CUDA context
- Apple ARM workers (worker-aarmd): MUST use Metal API for unified memory allocation
- Other workers: Define FFI boundaries in worker-specific specs
- All workers MUST own complete memory lifecycle (allocate → use → free)
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
- A single worker per GPU MUST be enforced (batch=1). Worker-specific memory policies (e.g., VRAM-only for worker-orcd) MUST be enforced by the respective worker implementation.
- Configuration SHOULD be minimal; sensible defaults MUST be provided for ports and timeouts.
---
### 3.2 Lab Mode (M2) (SYS-3.2.x)
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
### 3.3 Multi-GPU Mode (M4) (SYS-3.3.x)
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
- Memory accounting MUST handle per-GPU allocation and reserved headroom (VRAM for NVIDIA workers, unified memory for Apple ARM workers); preflight MUST fail when insufficient free memory across required GPUs.
- Cancellation and timeouts MUST apply per-job; multi-GPU workers MUST release all GPU allocations on job termination.
---
### 3.4 Multi-Node Mode (M4) (SYS-3.4.x)
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
- Scheduling SHOULD prefer locality and capacity signals (e.g., free memory/VRAM); policy MUST be observable via metrics and logs.
---
### 3.5 Platform Mode (M3+) (SYS-3.5.x)
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
## 4. System-Level Requirements [All Milestones]
### 4.1 Intelligence Boundary [M2] (SYS-4.1.x)
#### [SYS-4.1.1] Intelligence Centralization
The system MUST centralize ALL intelligent decisions in queen-rbee. Pool managers and workers MUST be dumb executors.
**Intelligent decisions** (queen-rbee only):
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
### 4.2 Smart vs Dumb Architecture [M2] (SYS-4.2.x)
#### [SYS-4.2.1] Smart vs Dumb Boundary
The system MUST enforce a strict smart/dumb boundary:
**Smart components** (queen-rbee only):
- MUST make ALL policy decisions (admission, scheduling, eviction, retry, timeout)
- MUST use configured policies (no hardcoded decisions)
- MUST decide actions based on aggregated state
**Dumb components** (pool manager, worker):
- MUST execute commands received without interpretation
- MUST report facts and state without filtering
- MUST NOT make policy decisions
- MUST perform operational cleanup only (not recovery decisions)
---
### 4.3 Component Separation [M0+] (SYS-4.3.x)
#### [SYS-4.3.1] Binary Separation
The system MUST implement three separate binaries: queen-rbee, pool-managerd, and worker-orcd. Each component MUST communicate via HTTP APIs only.
#### [SYS-4.3.2] No Direct Client→Worker Communication
Clients MUST NOT communicate directly with workers. All client requests MUST go through queen-rbee.
**Clarification**: The orchestrator directly calls worker endpoints to proxy/relay requests, but clients never communicate with workers directly. This maintains the control plane boundary.
---
### 4.4 State Propagation [M1+] (SYS-4.4.x)
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
### 4.5 Multi-Node Support [M4] (SYS-4.5.x)
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
### 5.1 Client → Orchestrator (Agentic API) [M2] (SYS-5.1.x)
#### [SYS-5.1.1] Task Submission
<!-- SECURITY AUDIT [auth-min team]: CRITICAL VULNERABILITY - No authentication mentioned in task submission endpoint. In Home Mode this may be acceptable (localhost), but Lab/Platform modes MUST enforce authentication. The spec does not explicitly require bearer token validation on POST /v2/tasks. Attack vector: Unauthenticated job submission can lead to resource exhaustion, quota bypass, and denial of service. RECOMMENDATION: Add explicit requirement that POST /v2/tasks MUST validate LLORCH_API_TOKEN via timing_safe_eq() in Lab/Platform modes. See SYS-9.1.1 for mode-specific auth requirements. -->
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
- `temperature` is OPTIONAL with default 0.7 for sampling; for test reproducibility, temperature SHOULD be set to 0.0 (greedy sampling for validation, not a product constraint)
- `seed` SHOULD be provided for determinism, otherwise orchestrator MAY supply one and MUST record it
- `priority` MUST be one of `interactive` or `batch`; unknown values MUST be rejected with 400
- Orchestratord MUST validate model existence and token budgets at admission; invalid requests MUST return 4xx with a stable error code
- Queue behavior is mode-dependent: Platform mode MAY reject jobs with 429 when capacity thresholds are exceeded; Home/Lab modes typically use unbounded queues with custom policies defined in Rhai scheduler
#### [SYS-5.1.3] SSE Streaming
<!-- SECURITY AUDIT [auth-min team]: CRITICAL VULNERABILITY - SSE endpoint lacks authentication and authorization checks. Attack vector: Job ID enumeration attack - attacker can iterate job IDs (job-001, job-002, etc.) to access other users' inference results, potentially leaking sensitive prompts and outputs. This violates tenant isolation in Platform Mode and privacy in Lab Mode. RECOMMENDATION: (1) SSE endpoint MUST validate bearer token, (2) MUST verify job_id belongs to authenticated tenant/user, (3) Use cryptographically random job IDs (UUIDv4) instead of sequential IDs to prevent enumeration. Add requirement: "Orchestrator MUST authorize SSE access: verify bearer token AND verify job_id ownership before streaming events." -->
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
- Event order MUST be: `queued` (orchestrator-level) → `started` (worker execution begins) → zero or more `token` → zero or more `metrics` (interleaved) → terminal (`end` or `error`)
- Exactly one terminal event MUST be emitted per job
- Note: Worker-to-orchestrator SSE streams omit the `queued` event as workers only emit execution-level events
**Specs**: `bin/queen-rbee-crates/agentic-api/.specs/00_agentic_api.md`
---
### 5.2 Orchestrator ↔ Pool Manager [M2] (SYS-5.2.x)
#### [SYS-5.2.1] Pool Registration
<!-- SECURITY AUDIT [auth-min team]: HIGH SEVERITY - Pool registration is a critical trust boundary. Current spec says "MUST be authenticated" but lacks specifics. Attack vectors: (1) Rogue pool registration - attacker registers malicious pool to intercept jobs and exfiltrate prompts/outputs, (2) Pool ID hijacking - attacker re-registers existing pool_id to redirect traffic, (3) Endpoint spoofing - attacker provides malicious endpoint URL. RECOMMENDATIONS: (1) Pool registration MUST use timing_safe_eq() for bearer token validation, (2) MUST validate endpoint URL is reachable and responds to health check before accepting registration, (3) MUST implement pool identity verification (certificate pinning or pre-shared pool secrets), (4) Re-registration MUST require explicit deregistration first OR cryptographic proof of pool ownership, (5) Endpoint URLs MUST be validated against SSRF attacks (no localhost, no private IPs unless explicitly allowed), (6) In Platform Mode, pool registration MUST verify EU residency claims (see SYS-9.2.2). Add explicit requirement: "Pool registration endpoint MUST validate bearer token using timing_safe_eq() and MUST verify endpoint reachability before accepting registration." -->
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
<!-- SECURITY AUDIT [auth-min team]: MEDIUM SEVERITY - Heartbeat endpoint is vulnerable to replay attacks and spoofing. Attack vectors: (1) Heartbeat replay - attacker captures legitimate heartbeat and replays it to keep a dead pool appearing alive, (2) Heartbeat spoofing - attacker sends fake heartbeats for pool_id they don't own to manipulate scheduling decisions, (3) State injection - attacker sends malicious GPU/worker state to trigger incorrect scheduling or resource exhaustion. RECOMMENDATIONS: (1) Heartbeat MUST validate bearer token using timing_safe_eq(), (2) MUST verify pool_id in URL matches pool_id in payload, (3) MUST verify pool_id matches authenticated pool identity (from registration), (4) Implement nonce or sequence number to prevent replay attacks, (5) Validate timestamp is within acceptable clock skew window (±30s) and reject old timestamps, (6) Validate GPU/worker state data against schema to prevent injection attacks. Add requirement: "Heartbeat endpoint MUST validate bearer token, verify pool_id ownership, and reject replayed or stale heartbeats (timestamp older than 2× heartbeat_interval)." -->
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
- Missing heartbeats beyond timeout MUST mark the pool unavailable for scheduling; timeout is calculated as `heartbeat_interval_ms × missed_heartbeat_threshold` (default: 15000ms × 3 = 45s)
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
### 5.3 Pool Manager ↔ Worker [M1] (SYS-5.3.x)
#### [SYS-5.3.1] Worker Ready Callback
<!-- SECURITY AUDIT [auth-min team]: HIGH SEVERITY - Worker ready callback is an internal trust boundary that can be exploited. Attack vectors: (1) Rogue worker registration - attacker spawns fake worker and registers with pool-managerd to intercept jobs, (2) Worker ID spoofing - attacker registers with worker_id of legitimate worker to hijack its jobs, (3) URI injection - attacker provides malicious URI to redirect inference requests, (4) VRAM inflation - attacker reports inflated vram_bytes to manipulate scheduling decisions. RECOMMENDATIONS: (1) Ready callback MUST be authenticated even for localhost (use shared secret between pool-managerd and worker), (2) Pool-managerd MUST verify worker_id matches the worker_id it assigned during spawn, (3) MUST verify URI is reachable and matches expected port range, (4) MUST validate model_ref matches the model_ref pool-managerd commanded worker to load, (5) MUST validate vram_bytes is within expected range for the model (detect inflation attacks), (6) Callback endpoint SHOULD bind to localhost only (not 0.0.0.0) to prevent external access. Add requirement: "Worker ready callback MUST include a spawn token (generated by pool-managerd at worker spawn time) to prove worker identity. Pool-managerd MUST validate spawn token using timing_safe_eq() before accepting ready callback." -->
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
### 5.4 Orchestrator → Worker (Direct) [M2] (SYS-5.4.x)
#### [SYS-5.4.1] Inference Execution
<!-- SECURITY AUDIT [auth-min team]: CRITICAL VULNERABILITY - Worker execute endpoint has NO authentication mentioned. Attack vectors: (1) Direct worker access - if worker ports are exposed (not localhost-only), attacker can bypass orchestrator and submit jobs directly to workers, bypassing admission control, quotas, and billing, (2) Prompt injection - attacker can submit malicious prompts directly to worker, (3) Resource exhaustion - attacker can flood worker with requests, (4) Job ID spoofing - attacker can submit jobs with arbitrary job_id to confuse tracking. RECOMMENDATIONS: (1) Worker execute endpoint MUST validate bearer token using timing_safe_eq() even if worker is localhost-bound (defense in depth), (2) Worker MUST verify job_id is not already in progress (prevent duplicate execution), (3) Worker MUST validate max_tokens is within acceptable range (prevent resource exhaustion), (4) Worker SHOULD only accept requests from orchestrator IP (IP allowlist), (5) Worker ports SHOULD bind to localhost only in single-node deployments, (6) In multi-node deployments, worker-orchestrator communication MUST use mTLS. Add requirement: "Worker execute endpoint MUST validate bearer token (LLORCH_WORKER_TOKEN) using timing_safe_eq() before accepting inference requests. Orchestrator MUST include this token in execute requests." -->
<!-- PERFORMANCE AUDIT [deadline-propagation team]: 🎯 EXECUTE ENDPOINT PERFORMANCE - This is THE hot path! Every optimization here multiplies across thousands of requests. CRITICAL PERFORMANCE REQUIREMENTS: (1) Worker MUST parse request in <1ms (use zero-copy JSON parsing, no allocations), (2) Worker MUST validate job_id is not duplicate in <100μs (use in-memory HashSet, not database), (3) Worker MUST check deadline BEFORE starting inference (if deadline already exceeded, return 504 immediately, don't waste GPU cycles), (4) Worker MUST use streaming response (chunked transfer encoding) to send first token ASAP (don't buffer entire response), (5) Worker MUST implement request timeout (if orchestrator disconnects, abort inference within 100ms). DEADLINE INTEGRATION: Worker MUST parse X-Deadline header, calculate remaining_time(), and abort if remaining < estimated_inference_time (based on max_tokens × per_token_latency). This is our CORE PROMISE: no wasted GPU cycles! 🚀 -->
**Inference execution**:
```
POST {worker_uri}/execute
{
  "job_id": "job-xyz",
  "prompt": "Hello world",
  "max_tokens": 100,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.1,
  "stop": ["\n\n", "END"],
  "seed": 42
}
Response: Capability-dependent protocol (see SYS-5.7.x)

Text generation (capability: text-gen):
  SSE stream: started → token* → metrics* → end/error

Other capabilities (image-gen, audio-gen, embedding, multimodal):
  See SYS-5.7.x for protocol specifications
```
**Advanced Generation Parameters** (M0+, text-gen capability):
- `temperature` (0.0-2.0): Temperature scaling for sampling (default: 1.0)
- `top_p` (0.0-1.0): Nucleus sampling - keep tokens with cumulative probability >= top_p (default: 1.0, disabled)
- `top_k` (0-vocab_size): Top-k sampling - keep only top k tokens by probability (default: 0, disabled)
- `repetition_penalty` (0.0-2.0): Penalize tokens that appear in generation history (default: 1.0, disabled)
- `stop` (array of strings): Stop generation when any sequence is matched, max 4 sequences, each max 32 tokens (default: [])
- `seed` (uint64): RNG seed for reproducible sampling (optional, auto-generated if omitted)
**Requirements** (text-gen capability):
- SSE stream event order MUST be: `started` → zero or more `token` → zero or more `metrics` (interleaved as needed) → terminal (`end` or `error`)
- Exactly one terminal event MUST be emitted per job (`end` on success, `error` on failure/cancel)
- Workers MUST include stable error codes; cancellation MUST use `CANCELLED`
- Orchestrator MUST propagate `X-Correlation-Id` to worker requests; workers SHOULD echo it in SSE metadata
- Orchestrator MUST enforce request timeouts per job policy and close SSE cleanly on timeout with a terminal event
**Multi-modality requirements**: See SYS-5.7.x for protocol requirements for other capabilities
#### [SYS-5.4.2] Cancellation
<!-- SECURITY AUDIT [auth-min team]: HIGH SEVERITY - Worker cancel endpoint lacks authentication and authorization. Attack vectors: (1) Unauthorized cancellation - attacker can cancel any job by guessing job_id, causing denial of service, (2) Job ID enumeration - attacker can probe for active jobs by attempting cancellation, (3) Resource manipulation - attacker can cancel high-priority jobs to manipulate scheduling. RECOMMENDATIONS: (1) Cancel endpoint MUST validate bearer token using timing_safe_eq(), (2) MUST verify job_id is currently assigned to this worker (prevent cross-worker cancellation), (3) Use cryptographically random job IDs to prevent enumeration, (4) Rate-limit cancel requests to prevent DoS. Add requirement: "Worker cancel endpoint MUST validate bearer token and verify job_id is assigned to this worker before processing cancellation." -->
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
- Deadline: orchestrator MUST enforce a cancellation deadline (default 5s) after which it treats the job as cancelled and closes client SSE
**Specs**: `bin/worker-orcd/.specs/00_worker-orcd.md`
---
### 5.5 Multi-Modality Streaming Protocols [M3+] (SYS-5.7.x)
#### [SYS-5.7.1] Protocol Selection
Orchestrator MUST select streaming protocol based on worker capability:
- **text-gen**: SSE protocol (existing, see SYS-5.4.1)
- **image-gen**: JSON response protocol (future)
- **audio-gen**: Binary stream protocol (future)
- **embedding**: JSON response protocol (future)
- **multimodal**: Mixed SSE protocol (future)

**Orchestrator awareness**:
- Orchestrator MUST receive worker capability and protocol metadata from pool manager (via heartbeat)
- Orchestrator MUST map job capability to expected protocol
- Orchestrator MUST relay worker responses using capability-appropriate strategy
- Orchestrator does NOT need to understand worker internals (CUDA, Metal, etc.), only protocol contracts

#### [SYS-5.7.2] Protocol Relay Strategies
**Text generation (SSE relay)**:
- Orchestrator MUST relay SSE events from worker to client
- MUST preserve event order (started → token* → metrics* → end/error)
- MUST add orchestrator metadata (correlation_id, queue_time) without altering payload

**Image generation (JSON relay)** [Future - M3+]:
- Orchestrator MUST call worker, receive JSON response
- MUST wrap JSON in SSE `result` event for client
- Response format: `{"images": [...], "metadata": {...}}`

**Audio generation (Binary relay)** [Future - M3+]:
- Orchestrator MUST receive binary stream from worker
- MUST base64-encode and wrap in SSE `audio` event for client
- Response format: `{"format": "mp3", "data": "base64...", "duration_ms": N}`

**Embedding (JSON relay)** [Future - M3+]:
- Orchestrator MUST call worker, receive JSON response (non-streaming)
- MUST wrap JSON in SSE `result` event for client
- Response format: `{"embedding": [...], "dimensions": N}`

**Multimodal (Mixed SSE relay)** [Future - M4+]:
- Orchestrator MUST relay mixed SSE events (token, image, audio)
- MUST preserve generation order
- Event types: started → (token|image|audio)* → end/error

#### [SYS-5.7.3] Worker Protocol Contract
Workers MUST implement protocol appropriate for their capability:
- Workers MUST advertise capability in ready callback and health endpoint
- Workers MUST implement `/execute` endpoint with capability-specific response format
- Workers MUST document protocol in worker-specific spec (e.g., `01_worker_orcd.md`)
- Protocol contracts MUST be defined in `bin/.specs/00_streaming_protocols.md` (future)

**Backward compatibility**:
- Existing text-gen workers (worker-orcd) continue using SSE protocol unchanged
- New capabilities are additive; no breaking changes to existing workers
---
### 5.6 Error Response Format [M0+] (SYS-5.5.x)
#### [SYS-5.5.1] Standard Error Response
<!-- SECURITY AUDIT [auth-min team]: MEDIUM SEVERITY - Error responses may leak sensitive information. Attack vectors: (1) Information disclosure - detailed error messages reveal system internals (GPU IDs, VRAM sizes, model paths), (2) Timing attacks - different error codes may have different response times, (3) Stack trace leakage - errors may include stack traces in development mode. RECOMMENDATIONS: (1) Error messages MUST NOT include raw tokens or bearer tokens (use token_fp6() for any token references), (2) Detailed error information (gpu_id, vram_bytes) SHOULD only be included for authenticated requests, (3) In Platform Mode, error details SHOULD be sanitized to prevent information leakage to untrusted clients, (4) Error responses MUST use timing-safe processing to prevent timing attacks, (5) Stack traces MUST be disabled in production, (6) Error messages MUST NOT reveal file paths, internal IPs, or system architecture details. Add requirement: "Error responses MUST sanitize sensitive information in Platform Mode. Internal details (gpu_id, vram_bytes, worker_id) MUST only be included for authenticated admin requests." -->
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
### 5.7 Correlation ID Propagation [M0+] (SYS-5.6.x)
#### [SYS-5.6.1] Correlation ID Requirements
<!-- SECURITY AUDIT [auth-min team]: LOW SEVERITY - Correlation ID handling has injection risks. Attack vectors: (1) Header injection - attacker provides malicious correlation ID with newlines or special characters to inject log entries or HTTP headers, (2) Log injection - correlation ID is logged without sanitization, allowing attacker to inject fake log entries, (3) XSS in dashboards - if correlation IDs are displayed in web dashboards without escaping, attacker can inject JavaScript. RECOMMENDATIONS: (1) Correlation IDs MUST be validated against a strict format (e.g., alphanumeric + hyphens only, max 64 chars), (2) Invalid correlation IDs MUST be rejected or replaced with generated ID, (3) Correlation IDs MUST be sanitized before logging (escape newlines, control characters), (4) Generated correlation IDs MUST use cryptographically random UUIDs (UUIDv4), (5) Correlation IDs MUST be HTML-escaped when displayed in dashboards. Add requirement: "Correlation IDs MUST match regex ^[a-zA-Z0-9-]{1,64}$. Invalid correlation IDs MUST be rejected and replaced with generated UUIDv4." -->
**Correlation**:
- `X-Correlation-Id` MUST be accepted from clients and propagated across orchestrator → pool-managerd → worker calls
- If the header is absent, orchestrator MUST generate a new correlation ID and propagate it downstream
- All logs and error responses MUST include the correlation ID
- SSE events SHOULD include correlation ID in metadata for client-side tracing
---
## 6. Component Architecture
### 6.1 Orchestratord (The Brain) [M2] (SYS-6.1.x)
**Binary**: `bin/queen-rbee/`  
**Port**: 8080 (default)  
**Role**: Centralized intelligence for all decisions
**Crates:**
- `scheduling` — Admission, queue, job tracking, worker selection, eviction
- `platform-api` — Marketplace federation facade
- `agentic-api` — Standard/home orchestrator API
- `pool-registry` — Track pool managers and state
- `streaming` — Protocol-aware relay (SSE, JSON, Binary) with metadata
- `task-cancellation` — Cancellation propagation
- `job-timeout` — Timeout enforcement
- `backpressure` — Queue backpressure handling
- `state-store` — Persistent state management (see SYS-6.1.3)
- `queue-optimizer` — Background optimization cron job (see SYS-6.1.4)
**Specs**: `bin/queen-rbee/.specs/00_queen-rbee.md`
---
#### [SYS-6.1.1] Orchestrator Intelligence
Orchestratord MUST implement ALL intelligent decision-making:
- MUST validate and admit requests before enqueue
- MUST manage bounded FIFO queue with Interactive/Batch priorities
- MUST select next job and target worker (combined scheduling decision)
- MUST command pool managers to start/stop workers
- MUST route inference requests to selected workers
- MUST relay worker responses to clients using capability-appropriate protocol (see SYS-5.7.x)
- MUST enforce timeout limits on jobs
- MUST propagate cancellation requests to workers
**Requirements**:
- All intelligent decisions (admission, scheduling, eviction, retry, timeout, cancellation) MUST occur in queen-rbee and MUST NOT be delegated
- Protocol relay to clients MUST preserve worker event order and MUST add orchestrator metadata without altering payload semantics
- Orchestrator MUST be aware of worker capabilities and their protocol contracts (received via pool manager metadata)
- Orchestrator MUST select relay strategy based on job capability (text-gen → SSE, image-gen → JSON, etc.)
- Orchestrator does NOT need to understand worker implementation details (CUDA, Metal, etc.), only protocol contracts
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
<!-- SECURITY AUDIT [auth-min team]: HIGH SEVERITY - SQLite database is a critical security boundary. Attack vectors: (1) SQL injection - if queries use string concatenation instead of parameterized queries, (2) File permission issues - if SQLite file has world-readable permissions, attackers can read all job data including prompts, (3) Backup exposure - if backups are not encrypted, they leak all historical data, (4) Concurrent access corruption - if WAL mode is not properly configured, database can be corrupted, (5) Prompt storage - spec mentions "prompt_hash" but doesn't prohibit storing raw prompts, which violates privacy. RECOMMENDATIONS: (1) ALL database queries MUST use parameterized queries (NEVER string concatenation), (2) SQLite file MUST have restrictive permissions (0600, owner-only read/write), (3) Database backups MUST be encrypted at rest, (4) Raw prompts MUST NEVER be stored in database (only prompt_hash), (5) Tenant IDs and session IDs MUST be indexed for efficient authorization queries, (6) Database file path MUST be validated to prevent path traversal attacks, (7) In Platform Mode, consider per-tenant database encryption. Add requirement: "SQLite database file MUST have 0600 permissions. ALL queries MUST use parameterized statements. Raw prompts MUST NOT be stored (only SHA-256 prompt_hash)." -->
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
**Specs**: `bin/queen-rbee-crates/state-store/.specs/00_state_store.md`
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
**Specs**: `bin/queen-rbee-crates/queue-optimizer/.specs/00_queue_optimizer.md`
---
#### [SYS-6.1.5] Programmable Scheduler
Orchestratord scheduler is designed as a **policy execution engine** that can run user-defined scheduling logic while maintaining a high-performance default.
**Deployment mode behavior**:
- **Platform Mode**: Uses immutable, built-in scheduler (written in Rhai) optimized for multi-tenant fairness, security, and SLA compliance. Queue policies (capacity limits, rejection thresholds) are defined in the platform scheduler.
- **Home/Lab Mode**: Users can write custom Rhai scripts or YAML configurations to define scheduling policies, including custom queue behavior (unbounded queues, custom eviction, etc.)
- **Web UI Mode**: Visual policy builder generates Rhai or YAML for non-programmers
**Language support**:
- **Rhai** (only): Rust-native scripting language with type safety, 0-indexed arrays, and built-in sandboxing. Lua is deprecated and no longer supported.
- **YAML** (declarative): Compiles to Rhai internally for simple rule-based policies
**Scheduler API**:
- Complete system state access (queue, pools, workers, GPUs, models, tenants)
- 40+ built-in helper functions (worker selection, GPU queries, quota checks, eviction)
- Preloaded model catalog at compile time
- Real-time pool state from heartbeats
- Sandboxed execution with 50ms timeout and memory limits
**Platform scheduler** (reference implementation):
- Location: `bin/queen-rbee-crates/scheduling/platform-scheduler.rhai`
- Immutable in platform mode, copyable in home/lab mode
- Optimized for priority-based scheduling, quota enforcement, resource utilization, and queue capacity management (may reject with 429 when thresholds exceeded)
**Specs**: 
- `bin/queen-rbee-crates/scheduling/.specs/00_programmable_scheduler.md` — Overall design and architecture
- `bin/queen-rbee-crates/scheduling/.specs/01_rhai_scheduler_runtime.md` — Rhai runtime environment and API reference
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
### 6.2 Pool-Managerd (Control Plane) [M1] (SYS-6.2.x)
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
- MUST query device state via platform-specific APIs (NVML for NVIDIA, Metal for Apple)
- MUST download and cache models as commanded
- MUST validate model-device compatibility before worker start (preflight)
- MUST spawn worker processes as commanded via worker adapters
- MUST update memory accounting when workers start/stop
- MUST send periodic heartbeats to queen-rbee (default 15s interval)
- MUST perform operational cleanup on worker failures (no retry decisions)
**Requirements**:
- Pool-managerd MUST NOT perform placement or retry decisions; it MUST execute orchestrator commands and report facts only
- Platform-specific APIs MUST be used for device queries (NVML for NVIDIA, Metal for Apple)
- Memory allocations MUST NOT be performed by pool-managerd (workers own memory lifecycle)
**Worker Adapter Support** (M3.5+):
- Pool-managerd MAY use worker adapters to support multiple worker implementations
- Default adapter MUST support bespoke CUDA worker (backwards compatibility)
- Additional adapters MAY be configured for llama.cpp, Apple Metal, image generation
- Adapter selection MUST be based on `worker_type` field in spawn request
- All workers MUST be normalized to common state format via adapter interface
- **Spec**: `bin/pool-managerd/.specs/10_worker_adapters.md` (POOL-1xxx)
---
#### [SYS-6.2.2] State Reporting
Pool-managerd MUST report facts, not decisions:
- MUST report GPU/device memory state (total, available, allocated)
- MUST report worker state (running, ready, failed)
- MUST report worker memory architecture (vram-only, unified)
- MUST report worker type and capabilities
- MUST report failures with context (exit code, error message)
- MUST NOT decide to retry or failover
**Heartbeat payload MUST include**:
- `pool_id` — Pool identifier
- `gpus` — Array of GPU states (id, total_vram_bytes, free_vram_bytes, device_name)
- `workers` — Array of worker states (id, status, model_ref, vram_bytes, gpu_id, uri, capabilities, protocol)

**Worker state fields** (M3+ for multi-modality support):
- `capabilities` — Array of capabilities (e.g., `["text-gen"]`, `["image-gen"]`) [REQUIRED]
- `protocol` — Streaming protocol (e.g., `"sse"`, `"json"`, `"binary"`) [REQUIRED]
- `worker_type` — Worker implementation type (e.g., `"bespoke-cuda"`, `"llamacpp"`) [OPTIONAL]

**Requirements**:
- Heartbeats MUST include device memory totals/allocated and worker states
- Heartbeats MUST include worker capabilities and protocol for each worker
- Missed heartbeats beyond timeout MUST mark the pool unavailable for scheduling
---
#### [SYS-6.2.3] Preflight Validation
<!-- PERFORMANCE AUDIT [deadline-propagation team]: 🎯 PREFLIGHT PERFORMANCE - Fail fast is our MANTRA! Preflight validation prevents wasted worker spawns. CRITICAL PERFORMANCE REQUIREMENTS: (1) Preflight MUST complete in <20ms (this is hot path, runs before every worker spawn), (2) NVML queries MUST be cached with 100ms TTL (don't query GPU state on every preflight, use recent heartbeat data), (3) Model file existence check MUST use stat() not open() (faster, no I/O), (4) VRAM calculation MUST be pessimistic (overestimate requirements by 10% to prevent OOM after spawn), (5) Preflight failures MUST return immediately with specific error code (don't retry, don't wait, just fail). OPTIMIZATION: Pool-managerd SHOULD maintain in-memory GPU state cache updated by heartbeats, preflight reads from cache (zero I/O overhead). TARGET: Preflight <20ms p99, zero false positives (never spawn worker that will OOM). Fast failure saves 60s of wasted worker startup! 🚀 -->
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
<!-- PERFORMANCE AUDIT [deadline-propagation team]: 🎯 HEARTBEAT PERFORMANCE - Heartbeats are overhead, minimize them! CRITICAL PERFORMANCE REQUIREMENTS: (1) Heartbeat generation MUST complete in <5ms (don't block pool manager operations), (2) Heartbeat payload MUST be <10KB (minimize network overhead, use compact JSON), (3) Heartbeat HTTP request MUST use persistent connections (avoid TCP handshake every 15s), (4) Orchestrator heartbeat processing MUST be <10ms (update in-memory state, don't block scheduling), (5) Heartbeat failures MUST NOT retry immediately (use exponential backoff to avoid thundering herd). OPTIMIZATION: (1) Pool-managerd SHOULD only send delta updates (only changed GPU/worker states, not full state every time), (2) Orchestrator SHOULD process heartbeats asynchronously (don't block admission/scheduling), (3) Use HTTP/2 for heartbeats (multiplexing reduces connection overhead). TRADE-OFF: 15s interval balances freshness vs overhead. Faster heartbeats = fresher state but more CPU/network. 15s is good default. TARGET: Heartbeat overhead <0.1% of pool manager CPU. 🚀 -->
Pool-managerd MUST send periodic heartbeats to queen-rbee:
- Default interval: 15 seconds (configurable)
- Payload MUST include: pool_id, timestamp, GPU states, worker states
- Heartbeat MUST be sent even if no state changes occurred
**Requirements**:
- Heartbeat failures MUST be logged but MUST NOT stop the pool manager
- Orchestratord MUST mark pools unavailable after missing N consecutive heartbeats (default: 3); total timeout = `heartbeat_interval_ms × N` (default: 15000ms × 3 = 45s)
- Clock skew detection SHOULD be implemented using timestamp comparison
---
#### [SYS-6.2.5] Operational Cleanup
Pool-managerd MUST perform operational cleanup on worker failures:
- MUST remove worker from registry
- MUST release VRAM accounting for failed worker
- MUST kill zombie processes
- MUST close file handles
- MUST report failure to queen-rbee with exit code and context
**Requirements**:
- Cleanup MUST be prompt (within 5 seconds of detection)
- Cleanup MUST NOT make retry decisions (queen-rbee decides)
- Cleanup failures MUST be logged and MAY trigger pool-level alerts
---
### 6.3 Worker Contract (Executor) [M0] (SYS-6.3.x)
**Default implementation**: `bin/worker-orcd/` (bespoke CUDA worker)  
**Extensibility**: Pool manager MAY support alternative worker implementations via adapter pattern (see `bin/pool-managerd/.specs/10_worker_adapters.md` POOL-1xxx spec)
**Worker contract requirements** (all workers MUST satisfy):
- [SYS-6.3.1] Worker self-containment (separate processes)
- [SYS-6.3.2] Worker isolation (no shared memory)
- [SYS-6.3.3] Tensor parallelism design (multi-GPU support)
- [SYS-6.3.4] Ready callback contract
- [SYS-6.3.5] Cancellation handling
- [SYS-6.3.6] HTTP API endpoints (/execute, /cancel, /health)
- [SYS-6.3.7] Memory reporting (VRAM for NVIDIA, unified for Apple)
- [SYS-6.3.8] Capability advertisement (text-gen, image-gen, audio-gen, embedding, multimodal)
- [SYS-6.3.9] Protocol contract implementation (SSE, JSON, Binary, Mixed SSE)
**Bespoke CUDA worker** (`worker-orcd`):
- **Binary**: `bin/worker-orcd/`
- **Port**: Dynamic (assigned by pool manager)
- **Role**: Execute inference on single GPU, or multi-GPU via single process for tensor parallelism
- **Memory architecture**: VRAM-only policy (NVIDIA CUDA only)
- **Implementation details**: See `bin/.specs/01_M0_worker_orcd.md`
- **Crates**: Integrated binary with CUDA modules, HTTP handlers, lifecycle management
**Apple ARM worker** (`worker-aarmd`) [Future - M3.5+]:
- **Binary**: `bin/worker-aarmd/` (Apple ARM daemon)
- **Port**: Dynamic (assigned by pool manager)
- **Role**: Execute inference on Apple Silicon using Metal backend
- **Memory architecture**: Unified memory (CPU and GPU share memory)
- **Implementation details**: See `bin/worker-aarmd/.specs/00_worker-aarmd.md` (future)
- **Platform**: macOS or Linux on Apple Silicon with Metal framework
**Other worker types**: See `bin/pool-managerd/.specs/10_worker_adapters.md` for llama.cpp, vLLM, and image generation workers.
---
#### [SYS-6.3.1] Worker Self-Containment
<!-- PERFORMANCE AUDIT [deadline-propagation team]: 🎯 SELF-CONTAINMENT PERFORMANCE BENEFIT - Excellent architectural choice! Process isolation enables zero-overhead testing and clean VRAM lifecycle. PERFORMANCE IMPLICATIONS: (1) Each worker owns its CUDA context = no context switching overhead, (2) Standalone testing means we can benchmark worker in isolation (pure GPU performance, no orchestrator noise), (3) Process boundaries enable clean shutdown = guaranteed VRAM cleanup in <5s, (4) One model per worker = no model switching overhead, optimal cache locality. RECOMMENDATION: Add explicit performance requirement: "Worker MUST be benchmarkable in standalone mode with <1% overhead vs direct CUDA inference (measure via microbenchmarks)." This proves our architecture doesn't add latency tax! 🚀 -->
All workers MUST operate as self-contained processes:
- MUST load exactly ONE model at startup (from disk to memory)
- MUST own memory allocation within its process context
- MUST execute inference requests received via HTTP
- MUST respond with capability-appropriate protocol (SSE for text-gen, JSON for image-gen/embedding, Binary for audio-gen)
- MUST monitor memory health (self-health checks)
- MUST report actual memory usage to pool manager on ready
**Worker-specific requirements**:
**Bespoke NVIDIA workers (worker-orcd)**:
- MUST allocate all model resources in VRAM only (no RAM fallback)
- MUST own VRAM allocation within its CUDA context
- MUST use memory-mapped I/O (mmap) for host I/O to avoid full RAM copies
- MUST copy model tensors to VRAM in bounded chunks (default 1MB) to prevent RAM spikes
- MUST support GGUF format (version 3 for MXFP4 tensor support)
- MUST validate model before loading (magic bytes, version, tensor count, VRAM fit)
- MUST detect model architecture from GGUF metadata (`general.architecture`)
- MUST support Llama-style architectures (RoPE, GQA, RMSNorm, SwiGLU) for Qwen/Phi-3 models
- MUST support GPT-style architectures (absolute pos embedding, MHA, LayerNorm, GELU) for GPT-OSS-20B
- MUST use architecture-specific model adapters to handle different model families (see Model Adapter in glossary)
- MUST support two tokenizer backends: `gguf-bpe` (Qwen/Phi-3) and `hf-json` (GPT-OSS-20B)
- MUST enforce UTF-8-safe SSE streaming (buffer partial multibyte sequences)
- MUST expose tokenizer metadata in health endpoint: `tokenizer_kind`, `vocab_size`
- Health endpoint MUST expose `quant_kind` field reflecting actually loaded quantization (MXFP4, Q4_K_M, Q4_0)
- **Spec**: `bin/.specs/01_M0_worker_orcd.md`
**Apple ARM workers (worker-aarmd)** [Future - M3.5+]:
- MUST use unified memory architecture (CPU and GPU share memory)
- MUST allocate model resources in unified memory accessible by both CPU and GPU
- MUST use Metal API for memory allocation and compute operations
- MUST support GGUF format with Metal-optimized loading
- MUST detect model architecture from GGUF metadata
- MUST use Metal Performance Shaders (MPS) for inference kernels
- MUST report unified memory usage (not separate VRAM)
- Health endpoint MUST expose `memory_architecture: "unified"`
- **Spec**: `bin/worker-aarmd/.specs/00_worker-aarmd.md` (future)
- **Platform**: macOS or Linux on Apple Silicon with Metal framework
**Other worker types**: See worker adapter spec for llama.cpp, vLLM, and image generation workers.
---
#### [SYS-6.3.2] Worker Isolation
Each worker MUST run in a separate OS process. Workers MUST NOT share memory or compute contexts.
**Requirements**:
- Each worker MUST have its own OS process
- Each worker MUST have its own compute context (CUDA context for NVIDIA, Metal context for Apple)
- Workers MUST NOT share memory pointers across processes
- Worker MUST own complete memory lifecycle (allocate → use → free)
**Rationale**: Memory allocations are per-process. Workers need isolated memory ownership.
**Worker-specific isolation**:
- **NVIDIA workers (worker-orcd)**: Each worker has its own CUDA context; VRAM allocations are per-process
- **Apple ARM workers (worker-aarmd)**: Each worker has its own Metal context; unified memory allocations are per-process
**Testing benefit**: Enables standalone worker testing (`worker-orcd --model X --gpu 0` or `worker-aarmd --model X`).
---
#### [SYS-6.3.3] Tensor Parallelism Design
**Tensor Parallelism Design** (M1+):
- A single worker process MAY use multiple compute devices for large models
- Worker maintains one compute context per device
- NOT multiple coordinated workers (maintains isolation principle)
- Pool-managerd tracks multi-device workers with device mask (e.g., GPUs 0,1,2)
**Requirements**:
- Multi-device workers MUST be tracked with device masks in pool-managerd
- Memory accounting MUST be per-device even for tensor-parallel workers
- Worker failures MUST release all device allocations atomically
**Worker-specific tensor parallelism**:
- **NVIDIA workers (worker-orcd)**: Use multiple CUDA contexts (one per GPU) for tensor parallelism
- **Apple ARM workers (worker-aarmd)**: Tensor parallelism not applicable (single unified memory pool)
---
#### [SYS-6.3.4] Ready Callback Contract
Worker MUST issue ready callback after initialization:
- HTTP server MUST start before the ready callback is sent (server-first)
- Callback MUST include `worker_id`, `model_ref`, `memory_bytes`, `memory_architecture`, and `uri`
- Callback endpoint is provided by pool-managerd at worker startup
**Requirements**:
- Ready callback to pool-managerd MUST include:
  - `worker_id`: Unique worker identifier
  - `model_ref`: Resolved model reference
  - `memory_bytes`: Actual memory usage (VRAM for NVIDIA, unified for Apple)
  - `memory_architecture`: Memory type (`vram-only` for NVIDIA, `unified` for Apple)
  - `uri`: Worker HTTP endpoint
  - `worker_type`: Worker adapter name (e.g., `bespoke-cuda`, `apple-metal`)
  - `capabilities`: Advertised capabilities (e.g., `["text-gen"]`)
- Callback failures MUST cause worker to exit with error code
- Pool-managerd MUST update memory accounting and mark worker ready atomically upon receiving callback
---
#### [SYS-6.3.5] Cancellation Handling
Worker MUST handle cancellation requests promptly:
- Upon receiving `POST /cancel`, worker MUST stop decoding
- Worker MUST free resources (memory, buffers)
- Worker MUST emit SSE `error` event with stable code `CANCELLED`
- Worker SHOULD return HTTP 202 to acknowledge cancellation
**Requirements**:
- Cancellation MUST be idempotent (repeated cancels for same job_id are safe)
- Worker MUST complete cancellation within deadline (default 5s)
- SSE stream MUST emit terminal `error` event with `CANCELLED` code
#### [SYS-6.3.6] HTTP API Endpoints
All workers MUST expose standard HTTP API endpoints:
- `POST /execute` — Execute inference request (response format depends on capability)
- `POST /cancel` — Cancel running inference
- `GET /health` — Health check and metadata (MUST include capability and protocol fields)

**Health endpoint requirements**:
- MUST include `capabilities` array (e.g., `["text-gen"]`, `["image-gen"]`)
- MUST include `protocol` field (e.g., `"sse"`, `"json"`, `"binary"`, `"mixed-sse"`)
- MAY include worker-specific metadata (memory architecture, quantization, etc.)
**Requirements**:
- Endpoints MUST conform to worker contract defined in worker adapter spec
- Health endpoint MUST expose worker metadata (model, memory usage, capabilities)
#### [SYS-6.3.7] Memory Reporting
Workers MUST report memory usage to pool manager:
- **NVIDIA workers (worker-orcd)**: Report VRAM usage via CUDA queries
- **Apple ARM workers (worker-aarmd)**: Report unified memory usage via Metal queries
- Memory reporting MUST be accurate and updated in real-time
#### [SYS-6.3.8] Capability Advertisement
Workers MUST advertise capabilities via ready callback and health endpoint:
- `text-gen` — Text generation (LLMs)
- `image-gen` — Image generation (Stable Diffusion, DALL-E)
- `audio-gen` — Audio generation (speech synthesis, music)
- `embedding` — Text embeddings
- `multimodal` — Vision-language models
Orchestrator uses capabilities for job routing, not worker types.
---
## 7. Data Flow & Interactions
### 7.1 Job Submission Flow [M2] (SYS-7.1.x)
<!-- PERFORMANCE AUDIT [deadline-propagation team]: 🎯 JOB SUBMISSION FLOW - THE CRITICAL PATH! Every millisecond here affects user experience. PERFORMANCE BREAKDOWN: (1) Admission validation MUST be <10ms (target from SYS-8.2.1), this includes model lookup, context length check, token budget check, (2) Enqueue MUST be <1ms (append to in-memory queue + SQLite insert), (3) Schedule MUST be <50ms (target from SYS-8.2.1), this includes Rhai script execution, pool state query, worker selection, (4) Dispatch MUST be <20ms (HTTP POST to worker + establish SSE connection), (5) First token MUST be <100ms (target from SYS-8.2.1). TOTAL TARGET: Admission to first token <180ms for hot path (worker already running). DEADLINE INTEGRATION: (1) Client SHOULD send X-Deadline header with absolute deadline, (2) Orchestrator MUST check deadline at admission (if already exceeded, return 504 immediately), (3) Orchestrator MUST check deadline before dispatch (if insufficient time remaining, return 504, don't waste worker), (4) Orchestrator MUST forward X-Deadline to worker for end-to-end enforcement. OPTIMIZATION: Use connection pooling, persistent HTTP connections, async I/O, zero-copy where possible. This is our SHOWCASE path! 🚀 -->
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
### 7.2 Worker Startup Flow [M1] (SYS-7.2.x)
<!-- PERFORMANCE AUDIT [deadline-propagation team]: 🎯 STARTUP FLOW PERFORMANCE - Worker startup is a CRITICAL latency path! 60s target is generous but we need to optimize every step. PERFORMANCE BREAKDOWN NEEDED: (1) Preflight validation should be <20ms (NVML queries are fast, don't add overhead), (2) Process spawn should be <100ms (fork+exec), (3) CUDA context init should be <1s (cudaSetDevice + cudaStreamCreate), (4) Model load to VRAM is the bottleneck (varies by model size, but we can optimize with hot-loading), (5) HTTP server start should be <100ms (bind port, start listener), (6) Ready callback should be <50ms (HTTP POST). OPTIMIZATION OPPORTUNITIES: (1) Pool-managerd SHOULD pre-load model files into RAM/page cache (hot-loading) to speed up worker startup by 10-50x, (2) Worker SHOULD use mmap for model loading (zero-copy from page cache to VRAM), (3) Worker SHOULD initialize CUDA context in parallel with HTTP server startup (overlap latencies), (4) Worker SHOULD emit progress events during model loading (0%, 25%, 50%, 75%, 100%) for observability. TARGET: Cold start <60s, hot start <5s. Let's make workers FAST! 🚀 -->
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
5. Worker: Initialize (worker-specific)
   - For worker-orcd (NVIDIA): Enforce VRAM-only, allocate VRAM
   - For worker-aarmd (Apple): Initialize Metal, allocate unified memory
   - model-lifecycle: Load model to designated memory
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
### 7.3 Worker Failure Flow [M1+] (SYS-7.3.x)
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
- Metrics and logs SHOULD record failure reason, retry attempt, and outcome;  MAY capture a failure timeline for reproduction
---
### 7.4 Cancellation Flow [M2] (SYS-7.4.x)
<!-- PERFORMANCE AUDIT [deadline-propagation team]: 🎯 CANCELLATION IS PERFORMANCE OPTIMIZATION! This is one of our FAVORITE features because it's all about NOT wasting cycles! CRITICAL PERFORMANCE REQUIREMENTS: (1) Client disconnect detection MUST be <100ms (poll SSE connection health every 50ms), (2) Orchestrator MUST issue cancel within <10ms of detecting disconnect (don't wait, act immediately), (3) Worker MUST detect cancel signal within <100ms (check cancellation flag every 10 tokens during inference), (4) Worker MUST stop inference within <50ms of detecting cancel (abort CUDA kernel, don't wait for token completion), (5) VRAM cleanup MUST complete within <1s (cudaFree all buffers), (6) Total cancellation latency MUST be <200ms (disconnect → inference stopped). DEADLINE INTEGRATION: Cancellation is deadline enforcement in action! When client disconnects, their deadline is implicitly NOW. Worker should also self-cancel when deadline exceeded (proactive waste prevention). METRICS: Track cancellation_latency_ms (p50, p95, p99) and cancelled_tokens_saved (how many GPU cycles we saved). This is our IMPACT metric! 🚀 -->
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
- Orchestrator MUST enforce cancellation deadline; after timeout it MUST treat job as cancelled and close client SSE
- Cancellation MUST use stable error code `CANCELLED` in SSE error event
- Job state MUST transition to "cancelled" in persistent store for audit trail
---
### 7.5 SSE Reconnection Flow [M2] (SYS-7.5.x)
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
### 8.1 Determinism [M0+] (SYS-8.1.x)
Content already defined in Section 2.3 (Foundational Concepts). Cross-reference: See SYS-2.3.1 through SYS-2.3.5 for complete determinism requirements.
**Summary**:
- System-level guarantees: Worker-specific memory architecture (VRAM-only for worker-orcd, unified memory for worker-aarmd), pinned engines, stable event ordering
- Model-level limitations: Engines and hardware may introduce non-determinism
- Best-effort approach with recording for reproducibility
---
### 8.2 Performance [M0+] (SYS-8.2.x)
#### [SYS-8.2.1] Latency Targets
<!-- PERFORMANCE AUDIT [deadline-propagation team]: 🎯 LATENCY TARGETS - YES! This section makes us VERY EXCITED! These are the numbers we live and breathe. CRITICAL ENHANCEMENTS NEEDED: (1) Targets are "SHOULD" but we need "MUST" for P0 paths - first token latency and queue admission are user-facing and MUST meet targets, (2) Missing deadline propagation overhead budget - each hop (orchestrator→pool→worker) adds latency, we need explicit budget allocation, (3) Missing client disconnect detection latency - how fast do we detect abandoned requests?, (4) Missing graceful degradation thresholds - at what latency do we start shedding load?, (5) Token generation rate is too vague - need per-model baselines with regression detection. RECOMMENDATIONS: (1) Strengthen first token latency to MUST <100ms for interactive priority jobs, (2) Add hop latency budget: orchestrator admission <10ms + scheduling <50ms + pool preflight <20ms + worker startup <60s + first token <100ms = total <60.18s, (3) Add client disconnect detection <100ms (poll SSE stream health every 50ms), (4) Add graceful degradation: if queue admission >50ms, start returning 503 with Retry-After, (5) Add per-model token rate baselines in config (e.g., llama-3.1-8b: 40-60 tok/s, llama-3.1-70b: 10-20 tok/s), emit alerts when actual rate <80% of baseline. DEADLINE PROPAGATION INTEGRATION: Every component MUST check remaining_time() before expensive work. If remaining < required, abort immediately with 504. This is our CORE RESPONSIBILITY and it MUST be in the spec! 🚀 -->
**Latency targets (measurement points)**:
- Queue admission SHOULD complete within 10ms measured from HTTP receive to enqueue decision
- Scheduling decision SHOULD complete within 50ms measured from job-ready to dispatch command issued (Rhai scheduler execution time included)
- Worker startup SHOULD complete within 60s measured from start command to ready callback receipt (note: includes model loading time which varies by model size; hot-loaded models from pool-managerd RAM cache may start faster)
- First token latency SHOULD be under 100ms measured from worker execute accept to first SSE `token` event
- Token generation rate SHOULD be within 20–100 tokens/sec depending on model; deviations MAY be acceptable with justification in metrics
<!-- PERFORMANCE AUDIT [deadline-propagation team]: 🎯 MISSING REQUIREMENT - Deadline enforcement latency targets! CRITICAL ADDITION NEEDED: "Deadline check overhead MUST be <1μs per check (measured via microbenchmarks). Components MUST check deadlines at these points: (1) Orchestrator: before enqueue, before dispatch, (2) Pool manager: before preflight, before spawn, (3) Worker: before model load, before inference, every 10 tokens during generation. Deadline exceeded MUST return 504 Gateway Timeout within <10ms of detection (no waiting for current operation to complete - abort immediately)." This is our PROMISE: we waste ZERO cycles on doomed work! 🚀 -->
**Requirements**:
- Latency targets are guidelines, not hard requirements
- Measurements MUST be instrumented via metrics (see SYS-10.1.x)
- Deviations SHOULD be logged and investigated
---
#### [SYS-8.2.2] Throughput and Limits
<!-- PERFORMANCE AUDIT [deadline-propagation team]: 🎯 THROUGHPUT OPTIMIZATION OPPORTUNITY - Unbounded queues are DANGEROUS for performance! CRITICAL CONCERNS: (1) Unbounded queue (-1) allows infinite backlog, leading to extreme latency for jobs at back of queue - a job enqueued when queue has 1000 items will wait HOURS, (2) No mention of queue admission backpressure - when should we return 503 instead of enqueuing?, (3) No mention of deadline-aware admission - we should REJECT jobs at admission if deadline already exceeded or insufficient time remains, (4) Batch=1 for M0 is correct but no mention of future batching impact on latency - batching increases throughput but adds latency, need clear trade-off documentation. RECOMMENDATIONS: (1) Add deadline-aware admission: "Orchestrator MUST reject jobs at admission if remaining_time(deadline) < estimated_execution_time (based on model + prompt length). Return 504 immediately, don't waste queue space.", (2) Add queue latency threshold: "If queue depth > 10 AND oldest job age > 60s, orchestrator SHOULD return 503 Service Unavailable with Retry-After for new admissions (backpressure).", (3) Add queue eviction policy: "When queue is full, orchestrator MUST evict jobs with soonest deadline expiry first (they're doomed anyway).", (4) Add throughput target: "Orchestrator MUST process admission requests at >100 req/s (p95 latency <10ms) to avoid becoming bottleneck." Performance is about saying NO to doomed work! 🚀 -->
**Throughput and limits**:
- Queue capacity MUST be configurable; default SHOULD be 100 jobs
- Worker concurrency for M0 MUST be 1 job per worker (batch=1)
- Multi-job batching MAY be enabled in M1+; when enabled, batching policies MUST be documented and observable via metrics
**Requirements**:
- Queue capacity of -1 indicates unbounded queue (infinite)
- Bounded queues MUST reject or evict when full per configured policy
- Throughput metrics MUST be emitted per component
---
### 8.3 Reliability [M1+] (SYS-8.3.x)
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
- Orchestratord MUST detect missed heartbeats from pools within a configured timeout and mark them unavailable for scheduling; timeout = `heartbeat_interval_ms × missed_heartbeat_threshold` (default: 15000ms × 3 = 45s)
- Retry policies MUST be configurable with exponential backoff parameters (see SYS-6.1.6)
**Requirements**:
- Worker failures MUST be detected within 5 seconds
- Pool failures MUST be detected within 3 missed heartbeats (default: 15000ms × 3 = 45s total)
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
- Test artifacts and incident analyses MAY include  to capture timelines and seeds for reproduction
**Requirements**:
- Logs MUST include correlation IDs for tracing
- Metrics MUST conform to SYS-10.1.x
-  SHOULD be generated per SYS-10.4.x
---
### 8.4 Scalability [M4] (SYS-8.4.x)
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
## 9. Security & Compliance [M3]
### 9.1 Authentication [M3] (SYS-9.1.x)
#### [SYS-9.1.1] Authentication by Deployment Mode
<!-- SECURITY AUDIT [auth-min team]: CRITICAL VULNERABILITY - Home Mode disables authentication entirely, creating a massive attack surface. Attack vectors: (1) Local privilege escalation - any process on the system can submit jobs, register pools, manipulate state, (2) Malware exploitation - malware running on the system has full access to orchestrator, (3) Accidental exposure - if user misconfigures bind address (0.0.0.0 instead of 127.0.0.1), system is exposed to network without authentication, (4) Multi-user systems - on shared systems, any user can access orchestrator. RECOMMENDATIONS: (1) Even in Home Mode, SHOULD support optional authentication via environment variable, (2) MUST enforce loopback binding (127.0.0.1) and MUST refuse to start if bind address is non-loopback without LLORCH_API_TOKEN set, (3) MUST validate bind address at startup using enforce_startup_bind_policy() from auth-min crate, (4) Consider adding a "development mode" flag that explicitly acknowledges security risks. CURRENT IMPLEMENTATION: auth-min crate already provides enforce_startup_bind_policy() which implements this protection. Spec MUST reference this requirement explicitly. Add requirement: "Home Mode MUST call enforce_startup_bind_policy() at startup. If bind address is non-loopback, LLORCH_API_TOKEN MUST be set or startup MUST fail." -->
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
<!-- SECURITY AUDIT [auth-min team]: HIGH SEVERITY - Lab Mode authentication is underspecified. Attack vectors: (1) Bearer token transmission in cleartext - if TLS is not used, tokens can be intercepted via network sniffing, (2) Weak token generation - spec doesn't specify token entropy requirements, (3) Token rotation - no mechanism for rotating tokens without downtime, (4) Token storage - no guidance on secure token storage in config files. RECOMMENDATIONS: (1) Lab Mode MUST use TLS (change SHOULD to MUST), (2) Bearer tokens MUST have minimum 256 bits of entropy (32 random bytes, base64-encoded), (3) Token validation MUST use timing_safe_eq() from auth-min crate, (4) Tokens MUST be stored in environment variables (not config files) to prevent accidental commits, (5) Support token rotation with grace period (accept old and new token during transition), (6) Document token generation command: openssl rand -base64 32. Add requirement: "Lab Mode bearer tokens MUST have ≥256 bits entropy. Token validation MUST use timing_safe_eq(). TLS MUST be used (not SHOULD)." -->
**Platform Mode (M2+)**:
- Multi-tenant marketplace, security is mandatory
- All client requests MUST be authenticated using HTTP bearer tokens
- Authentication CANNOT be disabled (security > performance)
- Pool-managerd registration MUST use bearer tokens or mTLS
- Orchestratord MUST validate tokens on every request (401/403 on failure)
- Inter-service calls SHOULD use mTLS for mutual authentication
- Audit logging MUST be enabled for all authenticated requests
<!-- SECURITY AUDIT [auth-min team]: CRITICAL VULNERABILITIES - Platform Mode has multiple critical gaps. Attack vectors: (1) Token management - no specification for token issuance, revocation, or expiry, (2) Tenant isolation - bearer tokens don't inherently carry tenant context, risk of cross-tenant access, (3) Token leakage - no guidance on token fingerprinting in logs, (4) Rate limiting - no mention of rate limiting to prevent brute force token guessing, (5) Token scope - no specification of token permissions/scopes, (6) mTLS is SHOULD not MUST for inter-service calls. RECOMMENDATIONS: (1) Platform Mode MUST use token_fp6() from auth-min crate for ALL token logging, (2) Bearer tokens MUST include tenant_id claim (use JWT with signature verification), (3) Token validation MUST use timing_safe_eq() for constant-time comparison, (4) Implement token expiry (max 24 hours) and refresh mechanism, (5) Rate limit authentication attempts (max 10 failures per IP per minute), (6) Inter-service calls MUST use mTLS (change SHOULD to MUST), (7) Implement token revocation list (check on every request), (8) Audit logging MUST use token_fp6() for actor identity (never raw tokens). Add requirement: "Platform Mode MUST use JWT bearer tokens with tenant_id claim and signature verification. Token validation MUST use timing_safe_eq(). Logging MUST use token_fp6(). Inter-service mTLS is MUST not SHOULD." -->
**Future provisions**:
- OAuth2/OpenID Connect MAY be added in future milestones (NOT REQUIRED for M0)
- API key authentication MAY be supported as alternative to bearer tokens
- If OAuth2/OIDC is configured, queen-rbee SHOULD validate audience/scope claims and enforce token expiry
---
### 9.2 EU Compliance (GDPR) [M3] (SYS-9.2.x)
#### [SYS-9.2.1] Data Residency
<!-- SECURITY AUDIT [auth-min team]: CRITICAL VULNERABILITY - Data residency enforcement is declarative only, no technical enforcement. Attack vectors: (1) False residency claims - pool-managerd can lie about EU location, (2) Data exfiltration - worker-orcd can transmit data to non-EU endpoints via DNS tunneling, HTTP callbacks, or model-embedded exfiltration, (3) Orchestrator compromise - if orchestrator is compromised, attacker can route jobs to non-EU pools, (4) Network routing - even if pool is physically in EU, network traffic may route through non-EU countries. RECOMMENDATIONS: (1) Implement cryptographic proof of location (e.g., GPS-signed attestations, datacenter certificates), (2) Worker-orcd MUST use network egress filtering to block non-EU IPs (allowlist EU IP ranges only), (3) Orchestrator MUST verify pool location via third-party geolocation service at registration, (4) Implement continuous monitoring of pool IP addresses for location changes, (5) Use EU-only DNS resolvers to prevent DNS-based exfiltration, (6) Audit all outbound network connections from workers (log destination IPs), (7) Consider using VPN/WireGuard to enforce EU-only network paths. Add requirement: "Pool registration MUST include cryptographic proof of EU location (datacenter certificate or GPS attestation). Worker-orcd MUST implement egress filtering to block non-EU destinations." -->
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
- MAY:  MAY include hashed or redacted payloads but MUST include correlation identifiers sufficient for auditing
---
#### [SYS-9.2.4] Audit Trail
**Audit trail**:
- MUST: All requests MUST be logged with correlation IDs propagated end-to-end
- SHOULD: Access to audit logs SHOULD be role-restricted and retention policies SHOULD comply with GDPR data minimization
**Compliance docs**: `.docs/.business/monetization.md`
---
### 9.3 Multi-Tenancy (Platform Mode) [M3] (SYS-9.3.x)
#### [SYS-9.3.1] Isolation Guarantees
<!-- SECURITY AUDIT [auth-min team]: CRITICAL VULNERABILITY - Tenant isolation is underspecified and has multiple attack vectors. Attack vectors: (1) VRAM side-channels - GPU memory is not cryptographically isolated, tenant A may be able to read tenant B's VRAM via timing attacks or memory scraping, (2) Model cache poisoning - if model cache is shared, tenant A can poison cache to affect tenant B's inference, (3) Worker reuse - if workers are reused across tenants, residual data in VRAM may leak, (4) Timing attacks - tenant A can infer tenant B's activity by measuring GPU utilization, (5) Log correlation - even with fingerprints, correlation attacks may link tenants, (6) Scheduler side-channels - tenant A can infer tenant B's job patterns via queue position changes. RECOMMENDATIONS: (1) Workers MUST NOT be reused across tenants (spawn fresh worker per tenant job), (2) VRAM MUST be zeroed after each job (cudaMemset to zero before worker exit), (3) Model cache MUST be per-tenant or use content-addressable storage with integrity checks, (4) Implement GPU memory encryption if available (NVIDIA Confidential Computing), (5) Add random delays to scheduling to prevent timing side-channels, (6) Logs MUST use tenant_fp6() (fingerprint tenant IDs) in addition to token_fp6(), (7) Metrics MUST NOT include tenant_id labels (use aggregated metrics only), (8) Consider dedicated GPU pools per tenant for high-security customers. Add requirement: "Workers MUST NOT be reused across tenants. VRAM MUST be zeroed (cudaMemset) before worker process exits. Model cache MUST be per-tenant or integrity-checked." -->
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
-  MUST include correlation and tenant identifiers and SHOULD avoid raw prompts unless debugging with redaction is explicitly enabled
---
#### [SYS-9.3.5] Observability and Billing
**Observability and billing** (Platform Mode only):
- **Prometheus metrics**: MAY omit or hash `tenant_id` to reduce cardinality and privacy risk
- **Billing/accounting logs**: MUST include plaintext `tenant_id` for usage tracking
- Usage accounting MUST be recorded per tenant for tokens generated, inference duration, and VRAM occupancy-time
- Billing data MUST be stored separately from observability metrics with appropriate access controls
- Home/Lab modes: Tenancy overhead is disabled (single-user assumption)
**Specs**: `bin/queen-rbee-crates/platform-api/.specs/00_platform_api.md`
---
## 10. Metrics & Observability [M-1+]
### 10.1 Metrics Contract [M0+] (SYS-10.1.x)
<!-- PERFORMANCE AUDIT [deadline-propagation team]: 🎯 METRICS FOR PERFORMANCE - YES! Metrics are how we PROVE our performance promises! CRITICAL ADDITIONS NEEDED: (1) Missing deadline enforcement metrics - we need to track how many requests we abort due to deadline exceeded, (2) Missing latency breakdown metrics - we need p50/p95/p99 histograms for every operation, (3) Missing client disconnect metrics - track how fast we detect and abort abandoned work, (4) Missing cancellation latency metrics - measure time from cancel request to inference stopped, (5) Missing first token latency - this is THE most important user-facing metric! REQUIRED METRICS FOR DEADLINE-PROPAGATION: (1) `deadline_enforced_total{component, reason}` - count of deadline aborts (reason: already_exceeded, insufficient_time, client_disconnect), (2) `deadline_exceeded_by_ms` - histogram of how late requests were when aborted, (3) `remaining_time_at_check_ms{component, checkpoint}` - histogram of remaining time at each deadline check (admission, dispatch, inference_start), (4) `deadline_check_duration_us` - histogram of deadline check overhead (<1μs target), (5) `first_token_latency_ms` - histogram from execute to first SSE token (p95 <100ms target), (6) `per_token_latency_ms` - histogram of inter-token timing (p95 <50ms target), (7) `cancellation_latency_ms` - histogram from cancel request to inference stopped (p95 <200ms target), (8) `client_disconnect_detection_ms` - histogram of disconnect detection time (p95 <100ms target). These metrics are our PERFORMANCE SCORECARD! 🚀 -->
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
### 10.2 Logging Standards [M-1+] (SYS-10.2.x)
#### [SYS-10.2.1] Log Format
**Format**:
- All components MUST emit JSON structured logs
- Log schemas SHOULD be stable and versioned to avoid breaking ingestion
**Levels**:
- Components MUST support standard levels: ERROR, WARN, INFO, DEBUG, TRACE
- Default level SHOULD be INFO in production; DEBUG/TRACE MAY be enabled temporarily for diagnostics
---
#### [SYS-10.2.2] Log Content
<!-- SECURITY AUDIT [auth-min team]: CRITICAL VULNERABILITY - Log content requirements are too weak. Attack vectors: (1) Token leakage - spec says "SHOULD avoid" but doesn't prohibit raw tokens in logs, (2) Prompt leakage - even with redaction, prompts may contain PII that leaks in truncated form, (3) Log injection - if correlation_id or other fields are not sanitized, attacker can inject fake log entries, (4) Bearer token leakage - no explicit prohibition on logging bearer tokens, (5) Tenant ID leakage - no guidance on fingerprinting tenant IDs. RECOMMENDATIONS: (1) Change "SHOULD avoid" to "MUST NEVER log" for raw bearer tokens, (2) Prompts MUST be hashed (SHA-256) not truncated, (3) ALL tokens (bearer, API, worker) MUST use token_fp6() from auth-min crate, (4) Tenant IDs SHOULD be fingerprinted in Platform Mode (tenant_fp6()), (5) Correlation IDs MUST be sanitized (alphanumeric only), (6) Implement log scrubbing in CI to detect accidental token leakage (grep for token patterns), (7) Error messages MUST use token_fp6() for any token references. Add requirement: "Logs MUST NEVER include raw bearer tokens. ALL token references MUST use token_fp6(). Prompts MUST be hashed (SHA-256) not stored raw. CI MUST enforce token leakage checks." -->
**Content**:
- Logs MUST include component, timestamp, level, correlation_id, and stable event codes for key actions (admission, schedule, dispatch, execute, cancel)
- Logs SHOULD avoid raw prompts/tokens unless explicitly enabled; when enabled, content MUST be redacted or truncated per policy
**Narration**:
- Human-readable narration fields MAY be included for developer ergonomics but MUST NOT replace structured fields
---
### 10.3 Correlation & Tracing [M0+] (SYS-10.3.x)
#### [SYS-10.3.1] Correlation ID Propagation
Content already defined in Section 5.6. Cross-reference: See SYS-5.6.1 for complete correlation ID requirements.
**Summary**:
- `X-Correlation-Id` MUST be accepted from clients and propagated across all service calls
- If absent, orchestrator MUST generate a new correlation ID
- All logs and error responses MUST include the correlation ID
- SSE events SHOULD include correlation ID in metadata
---
### 10.4 Proof Bundle Requirements [M-1+] (SYS-10.4.x)
#### [SYS-10.4.1] Proof Bundle Standard
** standard**:
- All automated test runs SHOULD produce  under `<crate>/.proof_bundle/<type>/<run_id>/`
-  MUST include an autogenerated header and MUST respect `LLORCH_RUN_ID` and `LLORCH_PROOF_DIR` when set
- Tests that rely on randomness MUST seed RNGs explicitly and record seeds in 
**References**:
-  standard: `libs/` crate
- Spec: `.specs/00_.md` (if exists in monorepo root)
**Requirements**:
-  MUST include: seeds, metadata, timestamps, correlation IDs
-  MAY include: transcripts, timelines, redacted payloads
-  SHOULD be deterministic (same inputs → same bundle structure)
---
## 11. Configuration
### 11.1 Orchestrator Config [M2] (SYS-11.1.x)
#### [SYS-11.1.1] Orchestrator Configuration Schema
```yaml
queen-rbee:
  bind: "0.0.0.0:8080"
  mode: "agentic"  # or "platform"
  queue:
    # Queue behavior is defined by the Rhai scheduler
    # Platform mode: scheduler may enforce capacity limits and reject with 429
    # Home/Lab mode: typically unbounded with custom policies
    capacity: -1  # -1 = unbounded (default for home/lab), positive integer for platform mode limits
  scheduling:
    scheduler_path: "platform-scheduler.rhai"  # path to Rhai scheduler script
    # Built-in algorithms available as Rhai helper functions:
    # - least-loaded, most-vram-free, round-robin
  eviction:
    # Two eviction types (both controlled by Rhai scheduler):
    # 1. Model eviction (hot-load eviction): Remove cached model files from pool-managerd RAM
    #    to free memory when models are no longer needed for quick worker startup
    # 2. Worker eviction: Stop worker processes to free VRAM for higher-priority jobs
    model_cache_policy: "lru"  # evict least-recently-used models from pool-managerd RAM cache
    worker_policy: "lru"       # stop least-recently-used workers to free VRAM
    vram_threshold: 0.9        # trigger worker eviction when VRAM > 90% utilized
  timeout:
    default_ms: 300000  # 5 minutes
    max_ms: 1800000     # 30 minutes
    cancellation_deadline_ms: 5000  # 5 seconds for worker to complete cancellation
  heartbeat:
    interval_ms: 15000  # pool-managerd heartbeat interval
    missed_threshold: 3  # mark pool unavailable after 3 missed heartbeats (45s total)
```
**Requirements**:
- Defaults MUST be applied when fields are omitted; values MUST be validated (e.g., `default_ms` ≤ `max_ms`)
- `mode` MUST be `agentic` or `platform`; unknown values MUST be rejected
- `queue.capacity`: -1 for unbounded (typical for home/lab), or positive integer (platform mode may use this for capacity-based rejection)
- Queue behavior (accept/reject/evict) is defined by the Rhai scheduler; platform mode scheduler may reject jobs with 429 when capacity thresholds are exceeded
- `scheduling.scheduler_path` MUST point to a valid Rhai script; built-in `platform-scheduler.rhai` is used if omitted in platform mode
- Timeouts MUST be enforced per job; values MAY be overridden per-request if allowed by policy
- `heartbeat.missed_threshold` determines pool unavailability timeout: `interval_ms × missed_threshold` (default: 45s)
---
### 11.2 Pool Manager Config [M1] (SYS-11.2.x)
#### [SYS-11.2.1] Pool Manager Configuration Schema
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
- `bind` MUST be a valid socket address; in home mode it SHOULD bind loopback unless explicitly overridden
- `pool_id` MUST be unique per orchestrator; collisions MUST be rejected at registration
- `orchestrator.url` MUST be a reachable HTTP endpoint; `heartbeat_interval_ms` MUST be a positive integer and SHOULD default to 15000
- `models.cache_dir` MUST be writable by pool-managerd; insufficient permissions MUST fail fast at startup
---
### 11.3 Worker Config [M0] (SYS-11.3.x)
<!-- PERFORMANCE AUDIT [deadline-propagation team]: 🎯 WORKER CONFIG PERFORMANCE - Configuration affects performance! MISSING PERFORMANCE-CRITICAL CONFIG: (1) No deadline enforcement config - worker needs to know if deadline checking is enabled, (2) No client disconnect polling interval - how often to check if orchestrator is still connected?, (3) No cancellation check interval - how often to check cancellation flag during inference?, (4) No graceful shutdown timeout - how long to wait before force-killing worker?, (5) No CUDA stream configuration - async vs sync operations?. RECOMMENDED ADDITIONS: Add flags: `--deadline-check-interval-tokens 10` (check deadline every N tokens), `--client-poll-interval-ms 50` (poll client connection every 50ms), `--cancellation-check-interval-tokens 10` (check cancel flag every N tokens), `--graceful-shutdown-timeout-ms 5000` (abort and exit within 5s), `--cuda-async-streams true` (use async CUDA for overlapped I/O). These configs let us TUNE performance per deployment! 🚀 -->
#### [SYS-11.3.1] Worker Configuration Schema
```bash
worker-orcd \
  --worker-id worker-abc \
  --model /models/llama-7b.gguf \
  --gpu-device 0 \
  --port 8001 \
  --callback-url http://pool:9200/v2/internal/workers/ready
```
**Requirements**:
- `--worker-id` MUST be unique per pool; collisions MUST be rejected
- `--model` MUST point to a readable file; startup MUST fail fast if missing or unreadable
- `--gpu-device` MUST refer to a valid CUDA device index present on the node
- `--port` SHOULD default to an ephemeral port if not provided; chosen port MUST be free at startup
- `--callback-url` MUST be reachable by pool-managerd and MUST use HTTP(S); ready callback MUST include `model_ref` and `vram_bytes`
---
### 11.4 Configuration Precedence [M0+] (SYS-11.4.x)
#### [SYS-11.4.1] Configuration Sources
**Configuration precedence** (highest to lowest):
1. Command-line arguments
2. Environment variables (prefixed with `LLORCH_`)
3. Configuration file (YAML)
4. Built-in defaults
**Requirements**:
- Higher precedence sources MUST override lower precedence sources
- Configuration validation MUST occur after all sources are merged
- Invalid configuration MUST fail fast at startup with clear error messages
- Configuration file path MAY be specified via `--config` flag or `LLORCH_CONFIG` environment variable
**Example environment variables**:
```bash
LLORCH_ORCHESTRATOR_BIND="0.0.0.0:8080"
LLORCH_ORCHESTRATOR_MODE="platform"
LLORCH_POOL_ID="pool-1"
LLORCH_WORKER_GPU_DEVICE="0"
```
---
## 12. Development Workflow
### 12.1 Spec-Driven Development (SYS-12.1.x)
#### [SYS-12.1.1] Development Process
**Process**: Spec → Contract → Tests → Code
**Workflow**:
1. Write spec (RFC-2119 normative requirements)
2. Define contracts (API, data structures)
3. Write tests (BDD, property tests, unit tests)
4. Implement code (guided by specs and tests)
**Docs**: `README_LLM.md`, `.docs/workflow.md`
---
### 12.2 Testing Strategy (SYS-12.2.x)
<!-- PERFORMANCE AUDIT [deadline-propagation team]: 🎯 TESTING STRATEGY MUST INCLUDE PERFORMANCE TESTS! Functional tests are great but we need PERFORMANCE VALIDATION! CRITICAL MISSING TEST TYPE: (1) Performance tests - measure latency, throughput, and resource usage, (2) Regression tests - detect performance degradation across commits, (3) Stress tests - validate behavior under load, (4) Deadline enforcement tests - verify deadline checking works end-to-end. REQUIRED PERFORMANCE TEST TYPES: (1) **Latency benchmarks** - measure p50/p95/p99 for all operations (admission, scheduling, first token, per-token, cancellation), (2) **Throughput benchmarks** - measure requests/second for admission, tokens/second for inference, (3) **Deadline enforcement tests** - verify deadline exceeded returns 504 within <10ms, verify insufficient time aborts before starting work, (4) **Cancellation latency tests** - measure time from cancel to inference stopped (<200ms target), (5) **Client disconnect tests** - measure detection time (<100ms target), (6) **Regression detection** - fail CI if p95 latency increases >10% vs baseline. Performance tests MUST run in CI on every commit! 🚀 -->
#### [SYS-12.2.1] Test Types
**Test types (scope and requirements)**:
- Unit tests MUST exist per crate to cover core functions and error paths
- Integration tests SHOULD validate cross-crate interactions and HTTP boundaries
- Contract tests MUST verify API conformance against the OpenAPI/contracts with stable IDs
- Property tests MUST enforce critical invariants (e.g., determinism, queue bounds, idempotent cancellation)
- BDD tests SHOULD derive from spec scenarios with stable identifiers and traceability to requirements
---
#### [SYS-12.2.2] Test Artifacts
**Artifacts and determinism**:
- All automated test runs SHOULD produce  under `<crate>/.proof_bundle/<type>/<run_id>/`
-  MUST include an autogenerated header and MUST respect `LLORCH_RUN_ID` and `LLORCH_PROOF_DIR` when set
- Tests that rely on randomness MUST seed RNGs explicitly and record seeds in 
**References**:
-  standard: `libs/` crate
- Spec: `.specs/00_.md` (if exists)
---
### 12.3 CI/CD Pipeline (SYS-12.3.x)
#### [SYS-12.3.1] CI Gates
<!-- PERFORMANCE AUDIT [deadline-propagation team]: 🎯 CI GATES MUST INCLUDE PERFORMANCE VALIDATION! We can't ship slow code! CRITICAL MISSING GATE: Stage 2.5 (Performance regression) - validate latency targets are met and no regressions vs baseline. REQUIRED PERFORMANCE GATE: **Stage 2.5 (Performance validation)**: (1) MUST run latency benchmarks for critical paths (admission <10ms p95, scheduling <50ms p95, first token <100ms p95, per-token <50ms p95), (2) MUST compare against baseline (previous commit or main branch), (3) MUST fail if p95 latency increases >10% without explicit waiver, (4) MUST validate deadline enforcement works (deadline exceeded returns 504 within <10ms), (5) MUST validate cancellation latency (<200ms p95), (6) SHOULD emit performance report to PR comments (before/after comparison). IMPLEMENTATION: Use criterion.rs for Rust benchmarks, store baseline in git, run benchmarks on every PR. Performance is not optional - it's a REQUIREMENT! 🚀 -->
**Gates (must-pass checks)**:
- Stage 0 (Spec hygiene) MUST pass link checks and ID stability; specs MUST use RFC‑2119 terms
- Stage 1 (Code quality) MUST pass `fmt` and `clippy` with no new warnings in changed code
- Stage 2 (Tests) MUST pass unit, integration, and property tests; flaky tests MUST be quarantined or fixed before merge
- Stage 3 (Contract compliance) MUST validate APIs against OpenAPI/contracts; breaking changes MUST be versioned and documented
- Stage 4 (Determinism suite) MUST verify determinism properties on supported targets
- Stage 5 (Metrics emission) SHOULD validate presence and type/label cardinality of required metrics; missing optional metrics MAY be tolerated with waivers
- Stage 6 (E2E acceptance) SHOULD pass scenario-based BDD tests for targeted milestones before release
**Planning references**:
- Roadmap documents SHOULD be maintained at `TODO.md` and `.plan/00_meta_plan.md`
---
## 13. Crate Dependency Graph
### 13.1 Dependency Overview
```
queen-rbee
├── scheduling (admission, queue, job-tracker, scheduler, eviction)
├── platform-api (marketplace facade)
├── agentic-api (standard API)
├── pool-registry (track pools)
├── streaming (SSE relay)
├── task-cancellation (cancel propagation)
├── job-timeout (timeout enforcement)
├── backpressure (backpressure handling)
├── state-store (persistent state management)
└── queue-optimizer (background optimization)
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
## 14. Milestone Roadmap
**Cross-Cutting Foundations** (All Milestones):
- **Logging**: `narration-core` enabled from M-1 for debugging and observability
- **Testing**: Unit tests and BDD tests required from M-1
- **Platform Awareness**: Architecture designed for platform mode from M-1; security deferred to M3
**M0 Scope Decision (2025-10-03)**:
- **Approach**: Performance Bundle Deferral (Hybrid)
- **Timeline**: M0 optimized to 4-5 weeks (from 6-8 weeks)
- **Deferred to M1**: Performance metrics, performance test suite, graceful shutdown, client disconnect detection, reproducible kernels validation, sensitive data handling
- **Removed**:  (entire concept removed from repo)
- **Reference**: See `bin/.specs/M0_RESOLUTION_CONTRADICTIONS.md` for full analysis
---
### 14.0 M-1: Foundation (Pre-M0)
**Goal**: Establish development infrastructure before first working system
**Scope**: Tooling, testing harness, shared crates foundation
**Deliverables**:
- `narration-core` logging infrastructure operational
- `test-harness/` structure established (BDD runner, e2e-haiku scaffold)
- Shared crate stubs: `auth-min`, `audit-logging`, `secrets-management`, `input-validation`
- CI pipeline skeleton (fmt, clippy, test runner)
- Development workflow documented
**Exit Criteria**:
- `cargo test` runs successfully across workspace
- BDD runner (`test-harness/bdd`) executes empty feature files
- Logging macros available and documented
**Note**:  removed from repo (M0 scope decision 2025-10-03)
**Status**: Foundation work
---
### 14.1 M0: Worker Haiku Test (v0.1.0) - HYBRID SCOPE
**Goal**: Prove a single worker can load a model in VRAM and execute inference functionally
**Scope**: `worker-orcd` binary only, standalone operation (performance validation deferred to M1)
**Scope Decision**: Performance Bundle Deferred (Hybrid Approach)
- **Timeline**: 4-5 weeks (optimized from 6-8 weeks)
- **Focus**: Functional correctness + critical safety features
- **Deferred to M1**: Performance validation, metrics, graceful shutdown
**Architecture**:
- **Performance > Security**: No authentication, localhost-only, minimal validation
- **Mode**: Home mode (single GPU, single worker, no orchestrator yet)
**Deliverables (Hybrid Scope)**:
- `worker-orcd` binary with:
  - Model loading into VRAM (CUDA FFI)
    - Memory-mapped I/O (mmap) for host I/O to avoid RAM copies
    - Chunked VRAM transfer (1MB chunks) to prevent RAM spikes
    - GGUF v3 format support (MXFP4 tensor blocks)
    - Architecture detection from GGUF metadata (`general.architecture`)
  - **Quantized-only execution** (Q4_K_M, MXFP4, Q4_0 - NO dequantization to FP32)
    - In-kernel dequantization to registers/shared memory
    - FP16 accumulation for all matmul results
    - FP16 KV cache precision
  - **Model adapters** (architecture-specific inference):
    - LlamaModelAdapter: RoPE, GQA, RMSNorm, SwiGLU (Qwen/Phi-3)
    - GPTModelAdapter: Absolute pos embedding, MHA, LayerNorm, GELU (GPT-OSS-20B)
  - HTTP inference API (`POST /v2/execute`, `GET /health`, `POST /cancel`)
    - **Advanced generation parameters**: temperature, top_p, top_k, repetition_penalty, stop sequences
  - SSE streaming (token-by-token output with UTF-8 boundary safety)
  - Haiku test endpoint or integration
  - **Dual tokenizer backends** (runtime selection):
    - `gguf-bpe`: Pure-Rust GGUF byte-BPE (Qwen/Phi-3)
    - `hf-json`: Hugging Face tokenizers crate (GPT-OSS-20B)
  - **Model load progress events** (0%, 25%, 50%, 75%, 100%) ← **CRITICAL** (user feedback)
  - **Memory residency verification** (periodic checks, worker-specific) ← **CRITICAL** (runtime safety)
  - **Memory OOM handling** (graceful error, not crash) ← **CRITICAL** (safety)
- Worker-orcd: VRAM-only enforcement (no RAM fallback, no UMA) - see `bin/.specs/01_M0_worker_orcd.md`
- Narration-core logging (basic events only, NO performance metrics)
- Temperature scaling (0.0-2.0 range)
**DEFERRED to M1** (Performance Bundle):
- ❌ Performance metrics emission (latency, throughput)
- ❌ Performance test suite
- ❌ Graceful shutdown endpoint (rely on SIGTERM)
- ❌ Client disconnect detection
- ❌ Reproducible kernels validation (implementation done, validation deferred)
- ❌ Sensitive data handling in logs
**M0 Reference Target Models**:
1. **Qwen2.5-0.5B-Instruct** (GGUF, Q4_K_M, 352 MB) — Primary bring-up & smoke test
2. **Phi-3-Mini (~3.8B) Instruct** (GGUF, Q4_K_M, 2.3 GB) — Stretch target within 24 GB
3. **GPT-OSS-20B** (GGUF, MXFP4, 12 GB) — Trend-relevant large model
**Execution Policy**: All models execute in quantized form (matches LM Studio/llama.cpp behavior). Per-tile dequantization in registers/shared memory during kernel execution; FP16 accumulation.
**Testing (Hybrid Scope)**:
- CUDA unit tests (functional only, NO performance tests):
  - GGUF header parsing (magic bytes, version, metadata extraction)
  - Architecture detection (Llama vs GPT)
  - VRAM allocation and residency checks
  - Kernel safety validation (no race conditions)
  - MXFP4 numerical correctness
- Rust unit tests:
  - HTTP request validation
  - SSE event ordering
  - Tokenizer conformance test vectors (20-30 pairs per model)
  - UTF-8 streaming boundary safety
- E2E haiku test in `test-harness/e2e-haiku/`:
  - Worker loads model (Qwen2.5-0.5B-Instruct)
  - Prompt includes current minute spelled out (e.g., "twenty-nine")
  - Worker generates haiku containing the minute word
  - Functional validation (reproducibility implementation done, validation deferred to M1)
  - Test temperature range 0.0-2.0 for product feature
  - Test advanced generation parameters (top_p, top_k, repetition_penalty, stop sequences)
  - Human verification (computer checks minute word presence, not haiku quality)
  - Basic test outputs (NO  - removed from repo)
- All three M0 models tested sequentially:
  - Qwen2.5-0.5B-Instruct (Q4_K_M, gguf-bpe tokenizer)
  - Phi-3-Mini (Q4_K_M, gguf-bpe tokenizer)
  - GPT-OSS-20B (MXFP4, hf-json tokenizer)
**DEFERRED to M1** (Performance Testing):
- ❌ Performance test suite (latency, throughput, memory leaks)
- ❌ Reproducible kernels validation
- ❌ Client disconnect detection tests
- ❌ Graceful shutdown tests
- ❌  emission (removed from repo)
**Exit Criteria (Hybrid Scope)**:
- Worker binary runs standalone: `worker-orcd --model <path> --gpu 0 --port 8001`
- Haiku test passes functionally with real GPU and model (no mocks) for Qwen2.5-0.5B
- All three M0 models load and execute successfully (sequential testing):
  - Qwen2.5-0.5B-Instruct: Llama-style architecture, Q4_K_M, gguf-bpe tokenizer
  - Phi-3-Mini: Llama-style architecture, Q4_K_M, gguf-bpe tokenizer
  - GPT-OSS-20B: GPT-style architecture, MXFP4, hf-json tokenizer
- Architecture detection works correctly from GGUF metadata
- Architecture-specific inference adapters execute correctly (Llama vs GPT pipelines)
- Model load progress events emit (0%, 25%, 50%, 75%, 100%) ← **CRITICAL**
- VRAM residency verification operational (periodic checks) ← **CRITICAL**
- VRAM OOM handling works (graceful error, not crash) ← **CRITICAL**
- VRAM usage tracked and logged (narration-core events)
- Quantized execution verified (no FP32 dequant on load, in-kernel dequant to registers)
- MXFP4 compute path validated (embeddings, attention, FFN, LM head)
- UTF-8 streaming validated (no mid-codepoint breaks)
- Tokenization works for both GGUF byte-BPE and tokenizer.json backends
- Advanced generation parameters work (top_p, top_k, repetition_penalty, stop sequences)
- Health endpoint exposes required fields: status, resident, quant_kind, tokenizer_kind, vocab_size
- Worker shuts down on SIGTERM (graceful shutdown endpoint deferred to M1)
**DEFERRED to M1** (Performance Exit Criteria):
- ❌ First token latency p95 <100ms
- ❌ Per-token latency p95 <50ms
- ❌ Health endpoint p99 <10ms
- ❌ Model loading time <60s
- ❌ Graceful shutdown <5s
- ❌ Zero memory leaks validation
- ❌ Client disconnect abort <100ms
- ❌  artifacts
**Non-Goals**:
- Orchestrator or pool manager (not yet)
- Multi-GPU or tensor parallelism
- Authentication or audit logging
- Persistent state or queue management
- Performance metrics/observability (deferred to M1)
- Performance test suite (deferred to M1)
- Graceful shutdown endpoint (deferred to M1)
- Client disconnect detection (deferred to M1)
-  (removed from repo)
**Status**: In progress (Hybrid Scope - 4-5 weeks)
---
### 14.2 M1: Pool Manager Lifecycle + M0 Performance Bundle (v0.2.0)
**Goal**: Pool manager lifecycle + complete M0 performance validation (deferred items from M0)
**M0 Deferred Items** (Performance Bundle - added to M1):
1. ✅ Performance metrics emission (worker_inference_duration_ms, worker_tokens_generated_total, latency metrics)
2. ✅ Performance test suite (first token latency, per-token latency, health endpoint, model loading time)
3. ✅ Graceful shutdown endpoint (POST /shutdown with 5s deadline)
4. ✅ Client disconnect detection (abort inference on SSE close)
5. ✅ Reproducible CUDA kernels validation (prove determinism works)
6. ✅ Sensitive data handling in logs (no raw prompts, only hashes)
7. ✅ Performance exit criteria validation (all targets from M0 spec)
**M1 Core Goal**: Pool manager can start/stop workers, hot-load models in RAM, and report pool state
**Scope**: `pool-managerd` binary with full worker lifecycle management
**Architecture**:
- **Performance > Security**: Minimal auth (optional bearer token), localhost or trusted network
- **Mode**: Home mode (single node, single pool manager, multiple workers possible)
**Deliverables**:
- `pool-managerd` binary with:
  - GPU inventory via NVML (read-only, no CUDA)
  - Worker lifecycle: spawn, monitor, terminate
  - Model provisioning: download from HuggingFace (`hf:` scheme) or local (`file:` scheme)
  - Model hot-loading: cache model files in RAM/page cache for fast worker startup
  - Preflight validation: check VRAM availability before spawning worker
  - State reporting API: pool state, worker status, VRAM usage
  - Heartbeat emission (periodic state report)
- Pool metrics: `pool_mgr_gpu_vram_total_bytes`, `pool_mgr_gpu_vram_allocated_bytes`, `pool_mgr_workers_total{status}`
- Operational cleanup: terminate zombie workers, free VRAM on worker crash
**Testing**:
- Unit tests for NVML queries, worker spawn/terminate, preflight validation
- Integration tests for model provisioning (mock HF download)
- BDD scenarios:
  - Pool manager starts worker successfully
  - Preflight fails when insufficient VRAM
  - Worker crash triggers cleanup
  - Hot-load cache speeds up second worker spawn
- E2E test in `test-harness/`:
  - Pool manager spawns worker
  - Worker passes haiku test
  - Pool manager reports accurate VRAM state
  - Pool manager terminates worker cleanly
**Exit Criteria**:
- Pool manager binary runs: `pool-managerd --config pool.yaml`
- Can spawn/terminate workers on command
- Preflight validation prevents VRAM overcommit
- Model hot-loading cache operational (measurable speedup)
- State reporting API returns accurate pool/worker/VRAM data
- Metrics emitted and scrapable
- Operational cleanup handles worker failures
**Non-Goals**:
- Orchestrator (not yet)
- Multi-node or multi-pool
- Intelligent scheduling decisions (pool manager is dumb executor)
- Authentication beyond optional bearer token
**Status**: Planned
---
### 14.3 M2: Orchestrator Scheduling (v0.3.0)
**Goal**: Orchestrator reads Rhai scheduler script and makes intelligent scheduling decisions
**Scope**: `queen-rbee` binary with full orchestration intelligence
**Architecture**:
- **Performance > Security**: Optional auth, focus on scheduling correctness
- **Mode**: Home mode or Lab mode (orchestrator + multiple pool managers)
**Deliverables**:
- `queen-rbee` binary with:
  - Agentic API: `POST /v2/tasks`, `GET /v2/tasks/{job_id}/events` (SSE)
  - Admission control: validate requests, check quotas
  - Queue management: priority queues (`interactive`, `batch`)
  - Programmable scheduler: Rhai script execution for scheduling decisions
  - Pool registry: track multiple pool managers
  - Heartbeat aggregation: collect pool state from all pool managers
  - Worker selection: choose pool/worker based on scheduler policy
  - Job dispatch: send inference requests to selected worker
  - SSE relay: stream worker output to client
  - Cancellation: propagate cancel requests to workers
  - Timeout enforcement: abort jobs exceeding time limits
  - Retry policy: retry failed jobs with exponential backoff
- Persistent state store (SQLite): queue state, job history, SSE checkpoints
- Orchestrator metrics: `orchestrator_queue_depth`, `orchestrator_admission_rate`, `orchestrator_scheduling_latency_ms`
**Testing**:
- Unit tests for admission, queue, scheduler, retry logic
- Property tests for deterministic scheduling (same inputs → same decisions)
- BDD scenarios:
  - Client submits task → queued → scheduled → executed → streamed
  - Scheduler selects worker based on Rhai policy
  - Job cancellation propagates to worker
  - Job timeout triggers abort
  - Worker failure triggers retry
  - SSE reconnection resumes from checkpoint
- Integration tests:
  - Orchestrator + pool manager + worker end-to-end
  - Multiple pool managers with load balancing
- E2E test in `test-harness/`:
  - Full system: orchestrator → pool manager → worker
  - Client submits haiku task via Agentic API
  - Orchestrator schedules to available worker
  - SSE stream delivers tokens to client
  - Metrics show scheduling latency and queue depth
**Exit Criteria**:
- Orchestrator binary runs: `queen-rbee --config orch.yaml`
- Agentic API accepts tasks and streams results
- Rhai scheduler executes custom policies (e.g., FIFO, priority, capacity-based)
- Multiple pool managers registered and heartbeating
- Scheduler selects workers based on VRAM availability and policy
- Cancellation and timeout work end-to-end
- Retry policy handles transient failures
- SSE reconnection preserves event order
- Persistent state survives orchestrator restart
- Metrics emitted and scrapable
**Non-Goals**:
- Multi-node orchestration (single orchestrator instance)
- Platform mode (no multi-tenancy or billing)
- Full authentication/authorization (minimal auth only)
- Audit logging (deferred to M3)
**Status**: Planned
---
### 14.4 M3: Security & Platform Readiness (v0.4.0)
**Goal**: Add security, auditing, and platform mode support
**Scope**: Security hardening, multi-tenancy, audit logging, performance review
**Architecture**:
- **Security > Performance**: Full authentication, authorization, audit logging
- **Mode**: Platform mode (multi-tenant marketplace)
**Deliverables**:
- `auth-min` crate: bearer token validation, timing-safe comparison, mode-aware auth
- `audit-logging` crate: structured audit events, tenant context, compliance-ready logs
- `secrets-management` crate: secure token storage, rotation, env-based loading
- Authentication enforcement:
  - Agentic API: validate `LLORCH_API_TOKEN` on all endpoints
  - Pool manager registration: validate bearer token or mTLS
  - SSE endpoint: authorize job ownership (prevent job ID enumeration)
- Authorization: tenant isolation, job ownership checks
- Audit logging:
  - Task submission, scheduling decisions, worker selection
  - Cancellation, timeout, retry events
  - Authentication failures, authorization denials
- Multi-tenancy:
  - Tenant context propagation (`X-Tenant-Id` header)
  - Per-tenant quotas (token budgets, rate limits)
  - Tenant-scoped metrics and logs
- Security hardening:
  - Cryptographically random job IDs (UUIDv4, prevent enumeration)
  - Input validation (prevent injection attacks)
  - TLS for network communication (Lab/Platform modes)
  - Secrets never logged or exposed in metrics
- Performance review:
  - Benchmark orchestrator throughput (tasks/sec)
  - Measure scheduling latency (p50, p95, p99)
  - Profile VRAM allocation overhead
  - Optimize hot paths (admission, queue, dispatch)
**Testing**:
- Unit tests for auth, audit, secrets management
- BDD scenarios:
  - Unauthenticated request rejected
  - Tenant A cannot access Tenant B's jobs
  - Audit log captures all security events
  - Quota enforcement prevents overuse
- Security tests:
  - Job ID enumeration attack blocked
  - Injection attacks fail validation
  - Timing attacks on token comparison fail
- Performance tests:
  - Orchestrator handles 100+ concurrent tasks
  - Scheduling latency < 10ms (p95)
  - VRAM allocation overhead < 5%
**Exit Criteria**:
- Authentication required in Lab/Platform modes (Home mode remains optional)
- Authorization prevents cross-tenant access
- Audit logs capture all security-relevant events
- Multi-tenancy enforced with quotas and isolation
- Security audit passes (no critical vulnerabilities)
- Performance benchmarks meet targets
- Platform mode operational (provider registration, federated routing)
**Non-Goals**:
- Multi-node orchestration (deferred to M4)
- Tensor parallelism (deferred to M4)
- Billing integration (business logic, not core system)
**Status**: Planned
---
### 14.5 M4: Multi-GPU & Multi-Node (v0.5.0)
**Goal**: Scale to multiple GPUs per node and multiple nodes
**Scope**: Tensor parallelism, cluster-wide orchestration, load balancing
**Features**:
- Tensor parallelism (single worker process using multiple GPUs)
- Multiple workers per node
- Multiple pool managers across nodes
- Cluster-wide orchestration
- Load balancing across nodes
**Status**: Future
---
### 14.6 M5: Platform Marketplace (v0.6.0)
**Goal**: Enable GPU provider ecosystem
**Scope**: Provider registration, federated routing, billing hooks
**Features**:
- Provider registration and discovery
- Federated routing (not nested orchestration)
- Billing and usage tracking hooks
- Provider-level SLIs and routing policies
**Status**: Future
---
## 15. Non-Goals / Out of Scope
**For worker-orcd (NVIDIA CUDA workers only)**:
- RAM fallback for model weights, KV cache, or activations — NOT SUPPORTED (VRAM-only policy for worker-orcd)
- CUDA Unified Memory (UMA) and zero-copy modes — NOT SUPPORTED for worker-orcd
- Disk swapping or spill for inference state — NOT SUPPORTED for worker-orcd
- CPU inference fallback — NOT SUPPORTED for worker-orcd
**System-wide**:
- Nested schedulers/orchestrators — NOT SUPPORTED; platform routing is federated, not nested (see §3.5)
**Note**: Other worker types (e.g., worker-aarmd for Apple ARM) have different memory architectures and constraints defined in their respective specs.
---
## 16. References
### 16.1 Specifications
**Component specs**:
- `bin/queen-rbee/.specs/00_queen-rbee.md` (ORCH-1xxx)
- `bin/pool-managerd/.specs/00_pool-managerd.md` (POOL-2xxx)
- `bin/worker-orcd/.specs/00_worker-orcd.md` (WORK-3xxx)
**Crate specs**: See individual crate `.specs/` directories
---
### 16.2 Documentation
- `README_LLM.md` — AI-optimized project overview
- `.docs/workflow.md` — Development workflow
- `.docs/.business/monetization.md` — Marketplace business model
- `.docs/test-case-discovery-method.md` — Testing approach
- `TODO.md` — Active roadmap
- `.plan/00_meta_plan.md` — Project plan
---
### 16.3 Contracts
- `/contracts/openapi/*.yaml` — OpenAPI specs
---
## 17. Appendices
### 17.1 Resolved Clarifications
This section documents clarifications that were originally inline HTML comments, now resolved:
1. **"No Direct Worker Communication" (formerly at SYS-4.3.2)**:
   - Clarified: Means "No Direct CLIENT→Worker Communication"
   - Orchestrator directly calls worker endpoints to proxy/relay requests
   - Clients never communicate with workers directly
2. **Retry Backoff Parameters (formerly at SYS-7.3.1)**:
   - Resolved: Full specification added at SYS-6.1.6
   - Initial delay: 1s, Multiplier: 2.0x, Max delay: 60s, Max attempts: 5
   - Jitter enabled by default
3. **Platform "Smart Router" (formerly at SYS-3.5.1)**:
   - Clarified: Platform orchestrator selects providers based on capacity/cost/region
   - Provider orchestrators make their own worker placement decisions
   - No nested scheduling
4. **Model Pinning SHOULD→MUST (formerly at SYS-2.1.2)**:
   - Resolved: Strengthened to MUST for determinism guarantee
   - `@rev` MUST be pinned to immutable commit SHA
   - `::file=...` MUST pin concrete artifact
5. **Temperature Parameter (formerly at SYS-5.1.2)**:
   - Clarified: OPTIONAL with default 0.7 for sampling
   - For deterministic inference, temperature SHOULD be set to 0 or omitted (engine-dependent)
   - Determinism tests assume temperature=0 for reproducibility
6. **Token Budgets (formerly at SYS-5.1.2, SYS-9.3.2)**:
   - Defined in Glossary (0.1)
   - Per-tenant quota for max tokens daily or per request
   - Enforced at admission time
---
### 17.2 Decision Log
**Key architectural decisions**:
1. **SQLite for State Store** (SYS-6.1.3):
   - Chosen over PostgreSQL for M0/M1 (single orchestrator)
   - ACID transactions, zero-ops, file-based backup
   - PostgreSQL reserved for M2+ HA scenarios
2. **Rhai for Programmable Scheduler** (SYS-6.1.5):
   - Chosen over Lua for Rust-native integration (Lua is deprecated)
   - Type safety, better error messages, 0-indexed arrays
   - Built-in sandboxing
   - All queue policies (capacity, rejection, eviction) defined in Rhai scheduler
3. **3-Level Traceability IDs** (SYS-X.Y.Z):
   - Chosen for scalability (1600+ line spec)
   - Hierarchical structure maps to document sections
   - Stable IDs survive content additions
4. **VRAM-Only Policy** (SYS-2.2.1):
   - No RAM/disk fallback for inference (models must fit entirely in VRAM during execution)
   - Enables test reproducibility (not determinism, which models cannot guarantee)
   - Simplifies VRAM accounting
   - Note: Pool-managerd MAY cache model files in RAM for hot-loading (faster worker startup), but workers MUST load models into VRAM for inference
5. **Smart/Dumb Boundary** (SYS-4.1.1, SYS-4.2.1):
   - All intelligence in queen-rbee
   - Pool managers and workers are dumb executors
   - Enables centralized policy control
---
### 17.3 Traceability Matrix
**Traceability from Spec IDs to Implementation**:
| Spec ID Range | Component | Implementation Path | Test Coverage |
|---------------|-----------|---------------------|---------------|
| SYS-2.1.x | Model Reference | `contracts/api-types/src/model_ref.rs` | Unit + Contract |
| SYS-2.2.x | VRAM-Only | `bin/worker-orcd/src/cuda/` | Property |
| SYS-2.3.x | Determinism | Cross-cutting | Property + BDD |
| SYS-4.x | System Requirements | Cross-cutting | BDD |
| SYS-5.x | API Contracts | `contracts/openapi/*.yaml` | Contract |
| SYS-6.1.x | Orchestratord | `bin/queen-rbee/src/` | Unit + Integration + BDD |
| SYS-6.2.x | Pool-Managerd | `bin/pool-managerd/src/` | Unit + Integration |
| SYS-6.3.x | Worker-Orcd | `bin/worker-orcd/src/` | Unit + Integration |
| SYS-7.x | Data Flows | Cross-cutting | BDD + Integration |
| SYS-8.x | Quality Attributes | Cross-cutting | Property + BDD |
| SYS-9.x | Security | Cross-cutting | Contract + Integration |
| SYS-10.x | Observability | Cross-cutting | Metrics tests |
| SYS-11.x | Configuration | `*/config.rs` | Unit |
**Test Catalog Reference**: `.docs/spec-derived-test-catalog.md`
---
**Version**: 0.1.0  
**Last Updated**: 2025-10-03  
**Status**: Living document (updated as system evolves)
---
**End of System Specification**
