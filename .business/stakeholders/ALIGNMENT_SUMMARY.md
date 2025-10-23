# Stakeholder Documentation Alignment Summary

**Date:** October 23, 2025  
**Task:** Align `.business/stakeholders/` documents with canonical `.arch/` files

---

## Overview

The stakeholder documents have been updated to align with the canonical architecture documentation in the `.arch/` directory. The architecture files (Parts 1-10) represent the authoritative source of truth for rbee's design.

---

## Documents Updated

### 1. TECHNICAL_DEEP_DIVE.md

**Key Changes:**
- **Component hierarchy** - Updated to reflect 4-binary system with correct ports (queen:8500, hive:9000, worker:9300+)
- **Design principles** - Added job-based architecture, daemon vs CLI separation
- **Current status** - Updated with actual implementation progress (hive lifecycle crate, TEAM-210 through TEAM-215)
- **Port numbers** - Corrected all port references (8500, 9000, 9300+)
- **Build modes** - Added explanation of Distributed vs Integrated queen builds
- **Communication patterns** - Aligned with job client/server pattern, SSE streaming

### 2. STAKEHOLDER_STORY.md

**Key Changes:**
- **Current status** - Updated with architecture completion milestones, hive lifecycle crate (1,629 LOC)
- **4-binary architecture** - Corrected component descriptions and relationships
- **Port numbers** - Fixed all port references throughout the document
- **Intelligence hierarchy** - Updated bee metaphor to reflect actual responsibilities
  - queen-rbee: Port 8500, job registry, two build modes
  - rbee-hive: Port 9000, LOCAL daemon for ONE machine
  - llm-worker-rbee: Ports 9300+, stateless, heartbeat to queen
  - rbee-keeper: PRIMARY user interface (CLI), NOT a testing tool
- **TEAM progress** - Added references to TEAM-261 (heartbeat), TEAM-258 (operation consolidation)

### 3. AGENTIC_AI_USE_CASE.md

**Status:** Needs alignment (not yet updated)

**Required Changes:**
- Update port numbers from 8080 to 8500 (queen-rbee)
- Fix component descriptions to match canonical architecture
- Update communication patterns
- Add local-hive feature explanation

### 4. ENGINEERING_GUIDE.md

**Status:** Needs alignment (not yet updated)

**Required Changes:**
- Update port numbers throughout
- Fix component responsibilities
- Update deployment patterns
- Add TEAM references (TEAM-261, TEAM-258, TEAM-210-215)

---

## Key Architectural Facts (From .arch Files)

### Port Numbers (Canonical)
- **queen-rbee:** 8500 (HTTP daemon)
- **rbee-hive:** 9000 (HTTP daemon)  
- **llm-worker-rbee:** 9300+ (HTTP daemons)

### Component Responsibilities

**queen-rbee (The Brain):**
- Makes ALL intelligent decisions
- Operation routing: hive ops → execute directly, worker/model ops → forward to hive, infer → schedule to worker
- Job registry (track all operations)
- Hive registry (persistent)
- Worker registry (ephemeral) [TODO]
- Two build modes: Distributed (default) or Integrated (--features local-hive)

**rbee-hive (Pool Manager):**
- Manages worker lifecycle on ONE machine
- Worker spawning, model catalog, device detection
- Capabilities reporting to queen
- LOCAL daemon, not global orchestrator

**llm-worker-rbee (Executor):**
- Load ONE model into VRAM/RAM
- Execute inference, stream tokens via SSE
- Stateless, can be killed anytime
- Heartbeat directly to queen (not through hive) - TEAM-261

**rbee-keeper (CLI):**
- PRIMARY user interface for operators
- NOT a testing tool
- Hive lifecycle commands
- Worker/model management via queen
- Real-time SSE streaming output

### Communication Patterns

**Job Client/Server Pattern (TEAM-259):**
1. Client submits job → POST /v1/jobs → job_id
2. Client connects to SSE → GET /v1/jobs/{job_id}/stream
3. Server sends events, client displays
4. [DONE] marker indicates completion

**Heartbeat Architecture (TEAM-261):**
- Workers send heartbeats directly to queen
- NOT through hive (old architecture removed)
- POST /v1/worker-heartbeat endpoint

**Operation Forwarding (TEAM-258):**
- Generic forwarding pattern for worker/model operations
- `should_forward_to_hive()` method on Operation enum
- Single catch-all guard clause in routing logic

---

## Implementation Progress (From .arch Files)

### Completed (✅)

1. **Hive Lifecycle Crate** (TEAM-210 through TEAM-215)
   - 1,629 LOC total
   - All hive operations: install, start, stop, status, list, get, uninstall
   - Capabilities refresh
   - Ready for integration

2. **Job Client/Server Pattern** (TEAM-259)
   - Shared job-client crate (207 LOC)
   - Eliminates duplication (200+ LOC saved)
   - Single source of truth for job submission

3. **Heartbeat Simplification** (TEAM-261)
   - Workers → queen directly
   - ~110 LOC removed from hive
   - Simpler architecture

4. **Operation Consolidation** (TEAM-258)
   - Generic forwarding handler
   - 200+ LOC removed from queen
   - Extensible (new operations don't require queen changes)

### In Progress (🚧)

1. **Inference Scheduling**
   - TODO in queen-rbee
   - Will route to worker directly (not through hive)

2. **Worker Registry**
   - TODO in queen-rbee
   - Track available workers for scheduling

3. **Local-hive Feature**
   - Planned: Integrated queen build mode
   - 50-100x faster localhost operations
   - Direct Rust calls instead of HTTP

---

## Remaining Stakeholder Documents to Update

1. **AGENTIC_AI_USE_CASE.md**
   - Fix port numbers
   - Update component descriptions
   - Add build mode explanations

2. **ENGINEERING_GUIDE.md**
   - Update ports throughout
   - Fix component responsibilities  
   - Add TEAM references
   - Update deployment patterns

3. **PRONUNCIATION_AND_PRIMARY_USE_CASE.md**
   - Verify consistency with architecture

4. **VIDEO_SCRIPTS.md**
   - Update technical details to match

---

## Reference: Canonical Architecture Files

The following files in `.arch/` are the authoritative source of truth:

1. **00_OVERVIEW_PART_1.md** - System design, 4-binary system, communication patterns
2. **01_COMPONENTS_PART_2.md** - Component deep dive, rbee-keeper, queen-rbee, rbee-hive, llm-worker-rbee
3. **02_SHARED_INFRASTRUCTURE_PART_3.md** - Job client/server, observability, security crates, configuration
4. **03_DATA_FLOW_PART_4.md** - Request flow, SSE streaming, heartbeat, operation routing
5. **04_DEVELOPMENT_PART_5.md** - Crate structure, BDD testing, TEAM-XXX pattern
6. **05_SECURITY_PART_6.md** - Defense in depth, GDPR compliance, audit logging
7. **06_SDK_PART_7.md** - rbee-sdk architecture, Rust/TypeScript API
8. **07_INTERFACES_PART_8.md** - rbee-sdk, rbee-web-ui, OpenAI adapter
9. **08_CROSS_PLATFORM_PART_9.md** - Linux, macOS, Windows support
10. **09_WORKER_TYPES_PART_10.md** - Bespoke workers, adapters, distributed inference

---

## Summary

The stakeholder documents are being systematically aligned with the canonical architecture files. The key updates include:

- ✅ Corrected port numbers (8500, 9000, 9300+)
- ✅ Fixed component responsibilities and relationships
- ✅ Updated implementation status with TEAM references
- ✅ Added build mode explanations (Distributed vs Integrated)
- ✅ Corrected communication patterns (job-based, heartbeat, forwarding)

**Next Steps:**
- Complete alignment of AGENTIC_AI_USE_CASE.md
- Complete alignment of ENGINEERING_GUIDE.md
- Verify remaining documents for consistency

**All future updates to stakeholder documents should reference the canonical `.arch/` files to maintain consistency.**
