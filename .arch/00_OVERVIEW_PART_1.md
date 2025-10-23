# rbee Architecture Overview - Part 1: System Design

**Version:** 1.0.0  
**Date:** October 23, 2025  
**Status:** Living Document  
**Audience:** Engineering teams, architects, security reviewers

---

## Table of Contents (All Parts)

1. **Part 1: System Design** (this document)
   - Mission & Philosophy
   - The Four-Binary System
   - Intelligence Hierarchy
   - Communication Patterns

2. **Part 2: Component Deep Dive**
   - rbee-keeper (CLI)
   - queen-rbee (Brain)
   - rbee-hive (Daemon)
   - llm-worker-rbee (Executor)

3. **Part 3: Shared Infrastructure**
   - Job Client/Server Pattern
   - Observability (Narration)
   - Security Crates
   - Configuration Management

4. **Part 4: Data Flow & Protocols**
   - Request Flow (End-to-End)
   - SSE Streaming
   - Heartbeat Architecture
   - Operation Routing

5. **Part 5: Development Patterns**
   - Crate Structure
   - BDD Testing
   - Character-Driven Development
   - Code Organization

6. **Part 6: Security & Compliance**
   - Defense in Depth
   - GDPR Compliance
   - Audit Logging
   - Threat Model

---

## Mission Statement

**rbee (pronounced "are-bee") is an OpenAI-compatible distributed LLM inference orchestration platform that enables developers to build AI infrastructure using their own hardware across home networks.**

### Core Value Propositions

1. **Independence** - Never depend on external AI providers
2. **Control** - Your models, your rules, never change without permission
3. **Privacy** - Code never leaves your network
4. **Cost** - Zero ongoing costs (electricity only)
5. **Multi-Node** - Use ALL your computers' GPU power

---

## Architectural Philosophy

### 1. Smart/Dumb Architecture

**Principle:** Clear separation between decision-making and execution.

```
queen-rbee (BRAIN)    → Makes ALL intelligent decisions
                        - Routing
                        - Scheduling
                        - Load balancing
                        - Resource allocation

llm-worker-rbee (EXECUTOR) → Dumb execution
                              - Load one model
                              - Execute inference
                              - Stream tokens
                              - Report status
```

**Why?**
- Testable components (each runs standalone)
- Scalable (add more executors without complexity)
- Maintainable (clear boundaries)
- Debuggable (failures isolated to component)

### 2. Process Isolation

**Principle:** Each worker runs in a separate process with isolated memory.

```
Worker 1 (Process A) → 8GB VRAM → llama-3-8b
Worker 2 (Process B) → 8GB VRAM → mistral-7b
Worker 3 (Process C) → 16GB RAM → qwen-0.5b (CPU)
```

**Why?**
- Memory safety (crash doesn't affect others)
- Resource accounting (clear VRAM ownership)
- Kill safety (can terminate without cleanup)
- Multi-model support (different models simultaneously)

### 3. Job-Based Architecture

**Principle:** All operations are jobs with unique IDs and SSE streams.

```
Client → POST /v1/jobs → job_id
Client → GET /v1/jobs/{job_id}/stream → SSE events
```

**Why?**
- Real-time feedback (SSE streaming)
- Job isolation (separate channels)
- Audit trail (every operation tracked)
- Cancellation support (kill by job_id)

### 4. Daemon vs CLI Separation

**Principle:** Daemons handle data plane, CLIs handle control plane.

```
DAEMONS (long-running, HTTP servers):
- queen-rbee (port 8500)
- rbee-hive (port 9000)
- llm-worker-rbee (ports 9300+)

CLIs (run on-demand, exit after command):
- rbee-keeper (user interface)
```

**Why?**
- Performance (daemons are fast, 1-5ms)
- UX (real-time SSE streaming)
- Resource efficiency (CLIs don't consume memory when idle)
- Clear separation (data vs control plane)

---

## The Four-Binary System

### Overview

rbee consists of **4 binaries** with clear responsibilities:

| Binary | Type | Port | Purpose | Status |
|--------|------|------|---------|--------|
| **rbee-keeper** | CLI | - | User interface, manages infrastructure | ✅ M0 |
| **queen-rbee** | Daemon | 8500 | Brain, routes all requests | 🚧 In Progress |
| **rbee-hive** | Daemon | 9000 | Worker lifecycle management | ✅ M0 |
| **llm-worker-rbee** | Daemon | 9300+ | LLM inference execution | ✅ M0 |

### Communication Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ USER (Human Operator)                                           │
└───────────────────┬─────────────────────────────────────────────┘
                    │ runs commands
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ rbee-keeper (CLI)                                               │
│ - hive install <alias>                                          │
│ - hive start <alias>                                            │
│ - worker spawn --model llama-3-8b --device GPU-0                │
│ - infer --prompt "Hello" --model llama-3-8b                     │
└───────────────────┬─────────────────────────────────────────────┘
                    │ POST /v1/jobs (Operation enum)
                    │ GET /v1/jobs/{job_id}/stream (SSE)
                    ↓
┌─────────────────────────────────────────────────────────────────┐
│ queen-rbee (HTTP Daemon - THE BRAIN)                           │
│ - Job registry (track all operations)                          │
│ - Operation routing (forwards to hive or worker)               │
│ - SSE streaming (real-time feedback)                           │
│ - Hive registry (track available hives)                        │
│ - Worker registry (track available workers) [TODO]             │
└─────────┬───────────────────────────────────┬───────────────────┘
          │ Hive operations                   │ Infer operations
          │ (WorkerSpawn, ModelDownload)      │ (direct to worker)
          ↓                                   ↓
┌─────────────────────────────────┐  ┌────────────────────────────┐
│ rbee-hive (HTTP Daemon)         │  │ llm-worker-rbee (HTTP)    │
│ - Worker lifecycle              │  │ - Load ONE model          │
│ - Model catalog                 │  │ - Execute inference       │
│ - Device detection              │  │ - Stream tokens via SSE   │
│ - Capabilities reporting        │  │ - Report health           │
└─────────────────────────────────┘  └────────────────────────────┘
```

---

## Intelligence Hierarchy

### Level 1: User Interface (rbee-keeper)

**Responsibility:** Provide user-friendly interface to infrastructure.

**Operations:**
- Hive lifecycle: `install`, `start`, `stop`, `status`, `uninstall`
- Worker lifecycle: `spawn`, `list`, `get`, `delete`
- Model management: `download`, `list`, `get`, `delete`
- Inference: `infer` (test inference)

**Key Insight:** This is NOT a testing tool, it's the PRIMARY user interface.

### Level 2: Orchestration Brain (queen-rbee)

**Responsibility:** Make ALL intelligent decisions.

**Decisions:**
1. **Routing:** Which hive handles which operation?
2. **Scheduling:** Which worker handles which inference?
3. **Load Balancing:** Distribute load across workers
4. **Health Management:** Track hive/worker availability
5. **Admission Control:** Accept or reject requests

**Key Insight:** Queen is the ONLY component that makes decisions.

### Level 3: Pool Management (rbee-hive)

**Responsibility:** Manage worker lifecycle on a single machine.

**Operations:**
- Worker spawning (detect GPU, spawn process)
- Worker registry (track local workers)
- Model catalog (track available models)
- Capabilities detection (GPU/CPU enumeration)

**Key Insight:** Hive is a LOCAL daemon, manages ONE machine.

### Level 4: Execution (llm-worker-rbee)

**Responsibility:** Dumb execution of inference.

**Operations:**
- Load model (into VRAM/RAM)
- Execute inference (generate tokens)
- Stream tokens (via SSE)
- Report health (heartbeat to queen)

**Key Insight:** Workers are STATELESS, can be killed anytime.

---

## Communication Patterns

### 1. Job Client/Server Pattern

**Purpose:** Unified pattern for job submission and SSE streaming.

**Flow:**
```rust
// 1. Client submits job
POST /v1/jobs
Body: { "operation": { "WorkerSpawn": { ... } } }
Response: { "job_id": "uuid" }

// 2. Client connects to SSE stream
GET /v1/jobs/{job_id}/stream
Response: text/event-stream

// 3. Server sends events
data: {"event": "worker_spawn", "message": "Spawning worker..."}
data: {"event": "worker_spawn", "message": "✅ Worker started"}
data: [DONE]
```

**Components:**
- `job-server` crate: In-memory job registry
- `job-client` crate: HTTP client with SSE support
- `rbee-operations` crate: Shared Operation enum

**Key Files:**
- `bin/99_shared_crates/job-server/src/lib.rs`
- `bin/99_shared_crates/job-client/src/lib.rs`
- `bin/99_shared_crates/rbee-operations/src/lib.rs`

### 2. Operation Forwarding Pattern

**Purpose:** Route operations from queen to appropriate service.

**Logic:**
```rust
match operation {
    // Hive operations → Handled directly in queen
    Operation::HiveInstall { .. } => execute_hive_install().await,
    Operation::HiveStart { .. } => execute_hive_start().await,
    Operation::HiveStop { .. } => execute_hive_stop().await,
    
    // Infer operations → Queen schedules, routes to worker
    Operation::Infer { .. } => {
        // TODO: Implement inference scheduling
        // 1. Query hive for available workers
        // 2. Select worker (load balancing)
        // 3. Direct HTTP to worker (not via job-client!)
        // 4. Stream tokens back to client
    }
    
    // Worker/Model operations → Forward to hive via HTTP
    op if op.should_forward_to_hive() => {
        hive_forwarder::forward_to_hive(&job_id, op, config).await
    }
}
```

**Key Insight:** Not all operations go through the same path!

### 3. Direct HTTP Pattern (Inference)

**Purpose:** Low-latency inference without job-server overhead.

**Flow:**
```
Client → queen-rbee (scheduling) → llm-worker-rbee (direct HTTP)
                                    POST /v1/inference
                                    ← SSE token stream
```

**Why Direct?**
- Performance (eliminate hop)
- Simplicity (no job-server on worker)
- Hot path optimization (inference is critical)

**Key Insight:** Queen circumvents hive for inference!

### 4. Heartbeat Pattern (Simplified in TEAM-261)

**Purpose:** Track worker availability for scheduling.

**Old Architecture (REMOVED):**
```
Worker → Hive (aggregation) → Queen
         POST /v1/heartbeat    POST /v1/heartbeat
```

**New Architecture (TEAM-261):**
```
Worker → Queen (direct)
         POST /v1/worker-heartbeat
```

**Benefits:**
- Simpler (no aggregation)
- Single source of truth (queen)
- Direct communication (no hop)

**Key Files:**
- `bin/10_queen_rbee/src/http/heartbeat.rs`
- `bin/30_llm_worker_rbee/src/heartbeat.rs`

---

## Queen Build Configurations

rbee-queen supports two build configurations to optimize for different deployment scenarios:

### Distributed Queen (Default)

**Build:** `cargo build --bin queen-rbee`

**Architecture:**
- Queen manages remote hives via HTTP
- All operations forwarded via job-client
- ~5-10ms overhead per operation
- Requires separate rbee-hive binary

**Use Cases:**
- Multi-machine deployments
- Cloud setups
- Distributed GPU clusters
- When queen runs on machine without GPUs

### Integrated Queen (local-hive feature)

**Build:** `cargo build --bin queen-rbee --features local-hive`

**Architecture:**
- Embedded hive logic for localhost operations
- Direct Rust calls for localhost (~0.1ms overhead)
- HTTP forwarding for remote hives (still available!)
- No separate rbee-hive binary needed for localhost

**Use Cases:**
- Single-machine setups
- Development environments
- Home labs with one GPU machine
- Performance-critical localhost operations

**Key Insight:** Even with `local-hive` feature, the queen can STILL manage remote hives via HTTP. It's the best of both worlds - fast local operations + distributed capability.

### Smart Recommendations

rbee-keeper will intelligently prompt you to rebuild queen with `local-hive` when:
- Installing hive on localhost
- Queen was built without local-hive feature
- Performance benefits would be significant (50-100x faster)

This ensures optimal performance while preserving deployment flexibility.

---

## Deployment Topologies

### Single Machine - Integrated (Recommended)

```
┌─────────────────────────────────────────────────────────┐
│ localhost                                               │
│                                                         │
│  rbee-keeper (CLI) → queen-rbee:8500 [local-hive]     │
│                           ↓ (direct Rust calls)        │
│                      llm-worker-rbee:9300               │
│                      llm-worker-rbee:9301               │
│                                                         │
│  ⚡ 50-100x faster localhost operations                │
│  📦 One binary instead of two                          │
└─────────────────────────────────────────────────────────┘
```

### Single Machine - Distributed (Optional)

```
┌─────────────────────────────────────────────────────────┐
│ localhost                                               │
│                                                         │
│  rbee-keeper (CLI) → queen-rbee:8500 → rbee-hive:9000 │
│                           ↓ (HTTP)                      │
│                      llm-worker-rbee:9300               │
│                      llm-worker-rbee:9301               │
│                                                         │
│  🌐 Distributed architecture on single machine         │
│  🔧 For specific use cases (testing, debugging)        │
└─────────────────────────────────────────────────────────┘
```

### Multi-Machine - Hybrid (Production)

```
┌──────────────────────────────────────────┐
│ Control Node                             │
│ - queen-rbee:8500 [local-hive]          │
│ - rbee-keeper (CLI)                     │
│ - llm-worker-rbee:9300 (local GPU)      │
│   ↓ (direct Rust calls for localhost)   │
└────────┬─────────────────────────────────┘
         │ HTTP (for remote hives)
         ├─────────────────────────────────────┐
         │                                     │
┌────────▼──────────────┐          ┌──────────▼────────────┐
│ GPU Node 1            │          │ GPU Node 2            │
│ - rbee-hive:9000      │          │ - rbee-hive:9000      │
│ - worker:9300 (GPU-0) │          │ - worker:9300 (GPU-0) │
│ - worker:9301 (GPU-1) │          │ - worker:9301 (GPU-1) │
└───────────────────────┘          └───────────────────────┘
```

### Multi-Machine - Pure Distributed

```
┌──────────────────────────┐
│ Control Node             │
│ - queen-rbee:8500        │
│ - rbee-keeper (CLI)      │
│   (no GPUs on this node) │
└────────┬─────────────────┘
         │ HTTP (all operations)
         ├─────────────────────────────────────┐
         │                                     │
┌────────▼──────────────┐          ┌──────────▼────────────┐
│ GPU Node 1            │          │ GPU Node 2            │
│ - rbee-hive:9000      │          │ - rbee-hive:9000      │
│ - worker:9300 (GPU-0) │          │ - worker:9300 (GPU-0) │
│ - worker:9301 (GPU-1) │          │ - worker:9301 (GPU-1) │
└───────────────────────┘          └───────────────────────┘
```

### Cloud-Native (Future)

```
┌────────────────────────────────────────────┐
│ Kubernetes Cluster                         │
│                                            │
│  queen-rbee (Deployment)                  │
│       ↓                                    │
│  rbee-hive (DaemonSet - one per node)    │
│       ↓                                    │
│  llm-worker-rbee (StatefulSet)           │
└────────────────────────────────────────────┘
```

---

## Key Architectural Decisions

### TEAM-261: Heartbeat Simplification

**Decision:** Workers send heartbeats directly to queen (not through hive).

**Rationale:**
- Simpler architecture (no aggregation)
- Single source of truth (queen knows all workers)
- Better for scheduling (direct worker state)

**Impact:**
- ~110 LOC removed from hive
- Queen is authoritative for worker state
- Hive is purely lifecycle management

**See:** `bin/.plan/TEAM_261_IMPLEMENTATION_COMPLETE.md`

### TEAM-258: Operation Consolidation

**Decision:** Consolidate worker/model operations into generic forwarding.

**Rationale:**
- DRY (Don't Repeat Yourself)
- Extensibility (add new operations without changing queen)
- Maintainability (single source of truth)

**Impact:**
- 200+ LOC removed from queen
- New operations don't require queen changes
- Clearer separation of concerns

**See:** Memory SYSTEM-RETRIEVED-MEMORY[72fe54ef-0d4c-4ee4-980c-3ad76fb122c0]

### TEAM-259: Job Client Consolidation

**Decision:** Extract job submission pattern into shared crate.

**Rationale:**
- Code duplication (same pattern in keeper and queen)
- Consistency (same behavior everywhere)
- Maintainability (fix bugs in one place)

**Impact:**
- 80 LOC removed from queen
- Reusable across all binaries
- Generic line handler (println vs NARRATE)

**See:** Memory SYSTEM-RETRIEVED-MEMORY[bafae083-ab3d-4d65-b712-5453114382e8]

---

## System Characteristics

### Performance

| Operation | Distributed Queen | Integrated Queen (local-hive) |
|-----------|-------------------|-------------------------------|
| Localhost hive operation | 5-10ms (HTTP) | ~0.1ms (direct call) |
| Remote hive operation | 5-10ms (HTTP) | 5-10ms (HTTP) |
| Worker spawn | 2-5s + overhead | 2-5s + overhead |
| Inference (first token) | 100-500ms | 100-500ms |
| Inference (subsequent tokens) | 20-50ms | 20-50ms |
| SSE streaming | < 1ms overhead | < 1ms overhead |

**Key Insight:** Integrated queen provides 50-100x faster localhost operations while maintaining same performance for remote operations.

### Scalability

| Component | Limit | Notes |
|-----------|-------|-------|
| Hives per queen | 100+ | Limited by network bandwidth |
| Workers per hive | 10-20 | Limited by GPU count |
| Concurrent inferences | 1000+ | Limited by worker count |
| SSE connections | 10000+ | Limited by file descriptors |

### Reliability

| Feature | Implementation | Status |
|---------|----------------|--------|
| Process isolation | Workers in separate processes | ✅ Done |
| Graceful shutdown | SIGTERM handling | 🚧 In Progress |
| Health checks | Heartbeat (30s interval) | ✅ Done |
| Auto-recovery | Worker respawn on crash | 📋 Planned |
| Circuit breaker | Retry with backoff | 📋 Planned |

---

## Next: Part 2 - Component Deep Dive

The next document covers detailed architecture of each binary:
- rbee-keeper CLI implementation
- queen-rbee brain architecture
- rbee-hive daemon design
- llm-worker-rbee execution model

**See:** `.arch/01_COMPONENTS_PART_2.md`
