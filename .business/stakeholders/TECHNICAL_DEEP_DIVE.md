# 🐝 rbee: Technical Deep Dive

> **A deep technical exploration of the bee-inspired AI orchestration platform** 🍯

**Version:** 0.1.0  
**Date:** 2025-10-10  
**Audience:** Technical stakeholders, architects, senior engineers  
**Website:** https://rbee.dev

---

## Executive Summary

This document provides a comprehensive technical overview of rbee's architecture for stakeholders who want to understand **how the system actually works** under the hood.

**Current Development Status (October 2025):**
- ✅ **31/62 BDD scenarios passing** (50% complete)
- ✅ Backend detection system operational (CUDA, Metal, CPU)
- ✅ Registry schema with backend capabilities
- 🚧 Lifecycle management in progress (TEAM-053)
- 🚧 Cascading shutdown implementation pending

**What You'll Learn:**
- Complete component architecture (4 binaries + their roles)
- End-to-end orchestration flow (40+ steps from request to response)
- BDD-driven development approach (TEAM-XXX pattern)
- Multi-modal protocol handling (text, images, audio, embeddings)
- Marketplace federation architecture
- Current implementation status and roadmap

---

## System Architecture Overview

### The Bee Metaphor (Component Hierarchy) 🐝

```
┌──────────────────────────────────────────────────────────┐
│ 👑🐝 queen-rbee (THE BRAIN)                              │
│ • Makes ALL intelligent decisions                        │
│ • Scheduling, routing, admission control                 │
│ • Rhai scripting engine for custom logic                 │
│ • HTTP daemon on port 8080                               │
└────────────────────┬─────────────────────────────────────┘
                     │ SSH control + HTTP inference
                     ↓
┌──────────────────────────────────────────────────────────┐
│### 2. 🍯🏠 rbee-hive (Hive Manager)                            │
│ • Executes commands from queen-rbee                      │
│ • Model catalog (SQLite)                                 │
│ • Worker lifecycle management                            │
│ • Health monitoring (30s heartbeat)                      │
│ • HTTP daemon on port 9200                               │
│ • SSH control (spawn workers, monitor health)          │
└────────────────────┬─────────────────────────────────────┘
                     │ Spawns processes
                     ↓
┌──────────────────────────────────────────────────────────┐
│ 🐝💪 [ai-type]-[backend]-worker-rbee (WORKER BEES)       │
│ • Load models into VRAM/memory                           │
│ • Execute inference (text, image, audio, embeddings)     │
│ • Stateless executors                                    │
│ • HTTP daemons on ports 8001+                            │
│ Examples:                                                 │
│   - llm-cuda-worker-rbee (LLM on CUDA)                   │
│   - llm-metal-worker-rbee (LLM on Apple Metal)           │
│   - sd-cuda-worker-rbee (Stable Diffusion on CUDA)       │
└──────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────┐
│ 🧑‍🌾🐝 rbee-keeper (USER INTERFACE)                        │
│ • Web UI (primary interface)                             │
│ • CLI (power users)                                      │
│ • Manages queen-rbee lifecycle                           │
│ • Configures SSH for remote machines                     │
└──────────────────────────────────────────────────────────┘
```

### Key Design Principles

**1. Smart/Dumb Separation**
- queen-rbee: Makes ALL decisions (smart)
- rbee-hive: Executes commands (dumb)
- worker-rbee: Generates tokens (dumb)

**2. Cascading Shutdown**
- When queen-rbee dies → ALL rbee-hives die (via SSH SIGTERM)
- When rbee-hive dies → ALL workers die (via HTTP shutdown)
- Result: No orphaned processes, no leaked VRAM

**3. Process Isolation**
- Each worker runs in separate process
- Each worker owns its memory context (CUDA context, Metal context, etc.)

**4. Protocol-Aware Orchestration**
- Text: SSE streaming
- Images: JSON response
- Audio: Binary stream
- Embeddings: JSON response

---

## The 4 Core Components

### 1. 👑🐝 queen-rbee (The Brain)

**Binary:** `bin/queen-rbee/`  
**Type:** HTTP Daemon, Port 8080  
**Language:** Rust

**Responsibilities:**
- Admission Control, Queue Management, Scheduling
- Worker Registry (SQLite)
- Hive Registry (SQLite with SSH details)
- SSE Relay (stream tokens from workers to clients)
- Rhai Scripting (user-defined routing logic)
- Marketplace Federation (route to external GPU providers)

**Key API Endpoints:**
```
POST   /v2/tasks                          # Submit inference job
GET    /v2/tasks/{job_id}/events          # SSE stream for job
POST   /v1/models/download                # Download model
GET    /v1/models/download/progress       # SSE download progress
POST   /v1/workers/spawn                  # Spawn worker
POST   /v1/workers/ready                  # Worker ready callback
GET    /v1/workers/list                   # List workers
```

**Database Schema (SQLite):**
```sql
CREATE TABLE models (
    id TEXT PRIMARY KEY,
    provider TEXT NOT NULL,  -- hf, file
    reference TEXT NOT NULL,
    local_path TEXT NOT NULL,
    size_bytes INTEGER
);
```

---

### 3. 🐝💪 [ai-type]-[backend]-worker-rbee (Worker Bees)

**Binaries:**
- `bin/llm-worker-rbee/` (LLM inference)
- `bin/sd-worker-rbee/` (Stable Diffusion - future)
- `bin/tts-worker-rbee/` (TTS - future)

**Type:** HTTP Daemon, Ports 8001+  
**Language:** Rust + Candle (ML framework)

**Responsibilities:**
- Model Loading (load into VRAM/memory)
- Inference Execution (generate tokens/images/audio)
- SSE Streaming (for text generation)
- State Management (loading, idle, busy)
- Ready Callback (notify rbee-hive when ready)

**Key API Endpoints:**
```
POST   /v1/execute                        # Execute inference
GET    /v1/ready                          # Readiness check
GET    /v1/loading/progress               # SSE loading progress
POST   /v1/cancel                         # Cancel inference
```

**State Machine:**
```
loading → idle → busy → idle
   ↓        ↓      ↓      ↓
   └────────┴──────┴──────┴──→ stopping → stopped
```

---

### 4. 🧑‍🌾🐝 rbee-keeper (User Interface)

**Binary:** `bin/rbee-keeper/`  
**Type:** CLI + Web UI  
**Language:** Rust (CLI) + Vue.js (Web UI)

**CLI Commands:**
```bash
# Setup (✅ IMPLEMENTED)
rbee-keeper setup add-node --name <name> --ssh-host <host>
rbee-keeper setup install --node <name>
rbee-keeper setup list-nodes

# Inference (✅ IMPLEMENTED)
rbee-keeper infer --node <name> --model <model> --prompt <prompt>

# Daemon (🚧 IN PROGRESS - TEAM-053)
rbee-keeper daemon start/stop/status

# Hive Management (🚧 PENDING - TEAM-053)
rbee-keeper hive start/stop/status --node <name>

# Worker Management (🚧 PENDING - TEAM-053)
rbee-keeper worker start/stop/list --node <name>
```

**Web UI Features:**
- Visual node management
- Model catalog browser
- Inference playground
- Worker status dashboard
- Rhai script editor

---

## Complete Orchestration Flow

### Cold Start Inference (No Workers Available)

```
PHASE 0: Setup (One-Time)
1. rbee-keeper setup add-node --name workstation
2. queen-rbee tests SSH connection
3. queen-rbee saves to beehives table
4. rbee-keeper setup install --node workstation
5. queen-rbee runs git clone + cargo build via SSH

PHASE 1: Job Submission
6. rbee-keeper infer --node workstation --model tinyllama
7. rbee-keeper → queen-rbee: POST /v2/tasks
8. queen-rbee queries beehives table for SSH details

PHASE 2: Hive Startup
9. queen-rbee → SSH: start rbee-hive daemon
10. rbee-hive starts HTTP daemon on port 9200

PHASE 3: Worker Registry Check
11. queen-rbee → rbee-hive: GET /v1/workers/list
12. rbee-hive responds: {"workers": []}  (empty)

PHASE 4: Model Provisioning
13. queen-rbee → rbee-hive: POST /v1/models/download
14. rbee-hive downloads from Hugging Face
15. rbee-hive streams progress via SSE
16. rbee-keeper displays progress bar
17. rbee-hive inserts model into SQLite catalog

PHASE 5: Worker Preflight
18. rbee-hive checks VRAM availability
19. rbee-hive checks backend availability (CUDA)

PHASE 6: Worker Startup
20. rbee-hive spawns: llm-cuda-worker-rbee --model ... --port 8001
21. Worker HTTP server starts on port 8001
22. Worker → rbee-hive: POST /v1/workers/ready
23. rbee-hive → queen-rbee: POST /v2/registry/workers/ready
24. queen-rbee registers worker in SQLite

PHASE 7: Model Loading
25. Worker loads model into VRAM (asynchronous)
26. rbee-keeper polls: GET /v1/ready
27. Worker streams loading progress via SSE
28. Worker completes loading → state = "idle"

PHASE 8: Inference Execution
29. rbee-keeper → worker: POST /v1/execute
30. Worker state: idle → busy
31. Worker generates tokens via Candle
32. Worker streams tokens via SSE
33. rbee-keeper displays tokens to stdout
34. Worker state: busy → idle

PHASE 9: Idle Timeout (5 minutes later)
35. rbee-hive monitors worker health every 30s
36. Worker idle for 5 minutes
37. rbee-hive → worker: POST /v1/admin/shutdown
38. Worker unloads model from VRAM
39. Worker exits cleanly
40. VRAM freed for other applications
```

---

## Multi-Modal Protocol Architecture

### Protocol Matrix

| Capability  | Protocol      | Content-Type           | Workers              |
|-------------|---------------|------------------------|----------------------|
| text-gen    | SSE           | text/event-stream      | llm-*-worker-rbee    |
| image-gen   | JSON          | application/json       | sd-*-worker-rbee     |
| audio-gen   | Binary        | audio/mpeg             | tts-*-worker-rbee    |
| embedding   | JSON          | application/json       | embed-*-worker-rbee  |

### Text Generation (SSE)

**Request:**
```json
POST /v1/execute
{
  "prompt": "write a story",
  "max_tokens": 100,
  "stream": true
}
```

**Response:**
```
Content-Type: text/event-stream

data: {"token": "Once", "index": 0}
data: {"token": " upon", "index": 1}
data: {"done": true, "total_tokens": 100}
```

### Image Generation (JSON)

**Request:**
```json
POST /v1/execute
{
  "prompt": "a cat on a couch",
  "width": 1024,
  "height": 1024
}
```

**Response:**
```json
{
  "images": [{
    "format": "png",
    "data": "base64...",
    "width": 1024,
    "height": 1024
  }]
}
```

---

## BDD-Driven Development

### Why BDD?

**Problem:** AI coders drift in huge codebases  
**Solution:** BDD keeps focus tight with executable specifications

**Benefits:**
- Executable specifications (Gherkin features)
- Clear acceptance criteria
- Prevents scope creep
- Human-readable
- Automated testing

**Current Status:**
- ✅ **31/62 scenarios passing** (50%)
- ✅ Setup scenarios working
- ✅ Registry operations working
- ✅ Model provisioning working
- 🚧 Lifecycle management pending (15+ scenarios blocked)
- 🚧 Exit code debugging needed (5+ scenarios)

### TEAM-XXX Development Style

**Pattern:**
1. Write Gherkin feature (`.feature` file)
2. Implement step definitions (Rust)
3. Run BDD tests (`bdd-runner`)
4. Iterate until green
5. Handoff to next team with summary

**Example from test-001.feature:**

```gherkin
Scenario: Happy path - cold start inference
  Given no workers are registered
  And node "workstation" is registered in rbee-hive registry
  When I run:
    """
    rbee-keeper infer --node workstation --model tinyllama
    """
  Then queen-rbee queries rbee-hive registry
  And queen-rbee establishes SSH connection
  And rbee-hive downloads the model
  And rbee-hive spawns worker
  And worker streams tokens via SSE
  And inference completes successfully
```

### Test Coverage (test-001.feature)

**Passing (31/62):**
- ✅ Setup scenarios (add node, install, list, remove)
- ✅ Registry operations
- ✅ Pool preflight checks
- ✅ Worker preflight checks
- ✅ Model provisioning (download, catalog)
- ✅ GGUF validation scenarios

**Failing (31/62):**
- ❌ Lifecycle management (15+ scenarios) - TEAM-053 priority
- ❌ SSH configuration (5+ scenarios)
- ❌ Exit code issues (5+ scenarios)
- ❌ Edge cases (4+ scenarios)
- ❌ Missing step definitions (2 scenarios)

**Target:** 54+ scenarios passing by end of M0

---

## Marketplace Federation

### Architecture: Federation, Not Nesting

**Our Approach:**
```
Platform queen-rbee (Marketplace Engine)
  ├─→ Provider A's queen-rbee (independent)
  ├─→ Provider B's queen-rbee (independent)
  └─→ Provider C's queen-rbee (independent)

Platform queen-rbee = Smart Router
```

### Provider Registration

```bash
# Provider runs their own queen-rbee
queen-rbee daemon &

# Register with marketplace
curl -X POST https://api.yourplatform.com/v2/platform/providers/register \
  -d '{
    "provider_id": "home-lab-123",
    "endpoint": "https://my-home-lab.example.com",
    "pricing": {"per_token": 0.0008},
    "capacity": {"total_gpus": 6},
    "geo": {"country": "NL", "region": "EU"}
  }'
```

### Task Routing

1. Customer submits task to platform
2. Platform queries registered providers
3. Platform selects best provider (pricing, SLA, latency)
4. Platform routes request to provider's queen-rbee
5. Provider orchestrates internally
6. Platform relays response to customer
7. Platform bills customer, pays provider (30-40% margin)

---

## Production Deployment

### Single-Node (Home Mode)

```bash
queen-rbee daemon &
rbee-hive daemon &
rbee-hive worker spawn cuda --model llama-7b --gpu 0
```

### Multi-Node (Lab Mode)

```bash
# Orchestrator
queen-rbee daemon &

# Remote nodes
ssh mac "rbee-hive daemon &"
ssh workstation "rbee-hive daemon &"

# Configure
rbee-keeper setup add-node --name mac --ssh-host mac.home.arpa
```

### Platform (Marketplace Mode)

```bash
queen-rbee daemon --mode platform &
# Providers register their queen-rbees
```

---

## Technical Specifications

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| First token latency | <100ms p95 | After model loaded |
| Per-token latency | <50ms p95 | Streaming |
| Admission latency | <10ms p95 | Queue admission |
| Scheduling latency | <50ms p95 | Worker selection |

### Scalability

- **Workers per hive:** 100+
- **Hives per queen:** 1000+
- **Concurrent jobs:** 10,000+
- **Tokens/second:** 1M+ (aggregate)

### Technology Stack

- **Language:** Rust (performance + safety)
- **ML Framework:** Candle (Rust-native)
- **Database:** SQLite (embedded)
- **HTTP:** Axum (async Rust web framework)
- **Scripting:** Rhai (embedded scripting)
- **Frontend:** Vue.js + TypeScript
- **Testing:** Cucumber (BDD)

---

## Development Workflow: TEAM-XXX Pattern

### How We Build rbee

**Approach:** BDD-driven development with AI coding teams

**Pattern:**
1. **Write Gherkin feature** - Define behavior in human-readable format
2. **Implement step definitions** - Write Rust code to execute steps
3. **Run BDD tests** - Execute `bdd-runner` to validate
4. **Iterate until green** - Fix failures, add missing implementations
5. **Handoff to next team** - Document progress, blockers, next priorities

**Example Handoff Chain:**
```
TEAM-051 → Port conflict resolution (✅ DONE)
TEAM-052 → Backend detection (✅ DONE)
TEAM-053 → Lifecycle management (🚧 IN PROGRESS)
TEAM-054 → Exit code debugging (📋 PLANNED)
```

### Current Sprint: TEAM-053

**Mission:** Implement lifecycle management + missing step definitions

**Priorities:**
1. **P0 (Critical):** Lifecycle commands (daemon/hive/worker start/stop/status)
2. **P0 (Critical):** Cascading shutdown implementation
3. **P1 (Important):** SSH configuration management
4. **P1 (Important):** Exit code debugging

**Expected Impact:**
- Phase 1: +2 scenarios (missing steps) → 33 passing
- Phase 2: +15 scenarios (lifecycle) → 48 passing
- Phase 3: +5 scenarios (SSH config) → 53 passing
- Phase 4: +1 scenario (exit codes) → 54 passing

**Timeline:** 10 days (2 weeks)

### Test Artifacts: Proof Bundles

**Every test run produces proof bundles:**
```
<crate>/.proof_bundle/<type>/<run_id>/
├── seeds.json         # RNG seeds for reproducibility
├── transcript.ndjson  # SSE events, HTTP requests
├── metadata.json      # Test metadata
└── result.txt         # Final output
```

**Benefits:**
- Deterministic testing (same seed → same output)
- Debugging aid (inspect exact events)
- Regression detection (compare bundles)
- CI/CD validation (automated checks)

**Standard:** `libs/proof-bundle` crate enforces consistent format across monorepo

---

## Current Implementation Status

### What's Working (✅)

**Infrastructure:**
- ✅ Backend detection (CUDA, Metal, CPU)
- ✅ Registry schema with backend capabilities
- ✅ Model catalog (SQLite)
- ✅ Model provisioning (Hugging Face download)
- ✅ Worker spawning (llm-cuda-worker-rbee)
- ✅ SSE streaming (token-by-token)
- ✅ HTTP APIs (queen-rbee, rbee-hive, worker)

**CLI Commands:**
- ✅ `rbee-keeper setup add-node`
- ✅ `rbee-keeper setup install`
- ✅ `rbee-keeper setup list-nodes`
- ✅ `rbee-keeper infer` (basic inference)
- ✅ `rbee-hive detect` (backend detection)

**Testing:**
- ✅ 31/62 BDD scenarios passing
- ✅ Unit tests passing (queen-rbee, rbee-hive, gpu-info)
- ✅ Proof bundle system operational

### What's In Progress (🚧)

**TEAM-053 (Current):**
- 🚧 `rbee-keeper daemon start/stop/status`
- 🚧 `rbee-keeper hive start/stop/status`
- 🚧 `rbee-keeper worker start/stop/list`
- 🚧 Cascading shutdown (queen-rbee → hives → workers)
- 🚧 SSH configuration management
- 🚧 Exit code debugging

**Expected Completion:** 2 weeks (mid-October 2025)

### What's Planned (📋)

**M0 Completion:**
- 📋 Worker cancellation endpoint
- 📋 Orphan cleanup automation
- 📋 Edge case handling (VRAM exhaustion, timeouts, etc.)
- 📋 54+ scenarios passing

**M1 (Q1 2026):**
- 📋 rbee-hive as HTTP daemon (persistent mode)
- 📋 Worker health monitoring (30s heartbeat)
- 📋 Idle timeout enforcement (5 minutes)
- 📋 Performance metrics emission

**M2 (Q2 2026):**
- 📋 queen-rbee HTTP daemon (orchestrator)
- 📋 Rhai scripting engine (user-defined routing)
- 📋 Web UI (visual management)

---

## Key Technical Decisions

### 1. BDD-Driven Development

**Why:** AI coders need tight focus to avoid drift in large codebases

**How:** Gherkin features define behavior, step definitions implement, tests validate

**Result:** 31/62 scenarios passing, clear progress tracking

### 2. Smart/Dumb Separation

**Why:** Centralize intelligence for easier debugging and customization

**How:** queen-rbee makes ALL decisions, hives/workers execute commands

**Result:** Easy to add Rhai scripting, marketplace federation

### 3. Process Isolation

**Why:** Prevent memory corruption, enable standalone testing

**How:** Each worker runs in separate process with own memory context

**Result:** Clean VRAM lifecycle, no cross-worker interference

### 4. Cascading Shutdown

**Why:** No orphaned processes, no leaked VRAM

**How:** queen-rbee death → hive death → worker death (via SIGTERM/HTTP)

**Result:** Deterministic cleanup, safe for testing

### 5. Protocol-Aware Orchestration

**Why:** Support multiple AI modalities (text, image, audio, embeddings)

**How:** queen-rbee knows capability → protocol mapping

**Result:** Unified API across modalities

---

---

## 🍯 Why the Bee Architecture Works

**Inspired by nature's most efficient distributed system:**

**👑🐝 Queen Bee (Centralized Intelligence)**
- Makes all strategic decisions
- Coordinates the entire colony
- Ensures harmony and efficiency

**🍯🏠 Hive (Resource Management)**
- Provides structure and storage
- Manages worker lifecycle
- Maintains the model catalog (honey stores)

**🐝💪 Worker Bees (Parallel Execution)**
- Execute tasks independently
- Process isolated (no interference)
- Scale horizontally (add more workers)

**🧑‍🌾🐝 Beekeeper (External Observer)**
- Manages the colony from outside
- Provides tools and configuration
- Monitors health and performance

**The result:** A system that scales like nature intended. 🐝

---

*Last Updated: 2025-10-10*  
*Based on: HANDOFF_TO_TEAM_053.md, TEAM_052_SUMMARY.md, test-001.feature, CRITICAL_RULES.md, COMPONENT_RESPONSIBILITIES_FINAL.md, MULTI_MODALITY_STREAMING_PLAN.md, monetization.md*  
*Built with 🍯 by AI engineering teams using Character-Driven Development*
