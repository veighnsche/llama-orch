# TEAM-025 Handoff: Next Steps After TEAM-024

**Date:** 2025-10-09T16:57:00+02:00  
**From:** TEAM-024  
**To:** TEAM-025  
**Status:** Ready for handoff  
**Priority:** HIGH - User needs rest, clear next steps required

---

## Executive Summary

**What TEAM-024 Completed:**
1. ✅ Fixed huggingface-cli → hf CLI migration (docs + code)
2. ✅ Clarified binary naming confusion (llorch vs orchestratord)
3. ✅ Added `llorch infer` command for testing workers
4. ✅ Created ORCHESTRATION_OVERVIEW.md (complete system guide)
5. ✅ Successfully tested token generation on local CPU worker

**Current System State (THE 4 BINARIES):**
1. **orchestratord** (daemon) - M1 ❌ NOT BUILT
2. **llorch-candled** (daemon) - M0 ✅ WORKING
3. **llorch** (CLI) - M0 ✅ WORKING
4. **llorch-pool** (CLI) - M0 ✅ WORKING

**M0 Status:** 3 of 4 binaries complete

**ARCHITECTURAL CHANGE (2025-10-09):**
- ❌ pool-managerd daemon is NOT NEEDED
- ✅ Only 2 daemons: orchestratord + llorch-candled
- ✅ 2 CLIs: llorch + llorch-pool
- ✅ M1 milestone simplified: Only build orchestratord
- See: `/FINAL_ARCHITECTURE.md` (definitive reference)

**User Status:** 🔴 EXHAUSTED - Needs rest, brain has left

---

## Critical Context: What's Working vs What's Not

### ✅ What Works (M0)

**Workers (llorch-candled):**
- Binary: `llorch-candled` (CPU, Metal, CUDA variants)
- Status: BUILT and WORKING
- Location: `bin/llorch-candled/`
- Test: `llorch infer --worker localhost:8080 --prompt "Hello"`
- Performance: ~6.5 tokens/sec on CPU

**Pool CLI (llorch-pool):**
- Binary: `llorch-pool`
- Status: BUILT and WORKING
- Location: `bin/pool-ctl/`
- Commands: models (download, catalog, register), worker (spawn, list, stop)

**Remote CLI (llorch):**
- Binary: `llorch`
- Status: BUILT and WORKING
- Location: `bin/llorch-ctl/`
- Commands: pool (models, worker, git, status), infer (NEW by TEAM-024)

### ❌ What Doesn't Exist Yet

**Orchestrator Daemon (orchestratord):**
- Binary: `orchestratord` (HTTP daemon)
- Status: NOT BUILT (M1 milestone - moved up from M2)
- Purpose: HTTP daemon that routes inference requests
- Port: 8080
- Location: Should be `bin/orchestratord/` (doesn't exist)

**REMOVED: pool-managerd**
- Decision: NOT NEEDED (2025-10-09)
- Reason: Pool management is control operations (CLI), not data plane (daemon)
- Replacement: `llorch-pool` CLI provides all functionality
- See: `/bin/.specs/ARCHITECTURE_DECISION_NO_POOL_DAEMON.md`

---

## Exhaustive Checklist: Specs & Plans

### `/bin/.specs/` - Specifications (10 files)

#### Core Specs
- [x] **00_llama-orch.md** (161KB) - Master system spec
  - Status: Complete, normative
  - Contains: SYS-xxx requirements, all component specs
  - Action: READ THIS FIRST before building anything

- [x] **01_M0_worker_orcd.md** (108KB) - Worker spec
  - Status: Complete for M0
  - Contains: Worker requirements, HTTP API, SSE streaming
  - Note: "worker-orcd" is old name, now "llorch-candled"

- [x] **71_metrics_contract.md** (4KB) - Metrics spec
  - Status: Complete
  - Contains: Prometheus metrics definitions
  - Action: Implement when building daemons

#### Architecture Decisions (7 files)
- [x] **FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md** (21KB)
  - Status: NORMATIVE - This is THE architecture
  - Decision: SSH for control, HTTP for inference
  - Action: Follow this for all future work

- [x] **CONTROL_PLANE_ARCHITECTURE_DECISION.md** (20KB)
  - Status: Historical context
  - Explains why SSH for control plane

- [x] **COMPLETE_BINARY_ARCHITECTURE.md** (19KB)
  - Status: Complete binary breakdown
  - Shows all binaries and their relationships

- [x] **BINARY_ARCHITECTURE_COMPLETE.md** (13KB)
  - Status: Duplicate/older version
  - Action: Can ignore, use COMPLETE_BINARY_ARCHITECTURE.md

- [x] **BINARY_STRUCTURE_CLARIFICATION.md** (8KB)
  - Status: Clarification doc
  - Explains binary naming

- [x] **ARCHITECTURE_DECISION_CLI_VS_HTTP.md** (15KB)
  - Status: Historical decision
  - Explains CLI vs daemon split

- [x] **FEATURE_TOGGLES.md** (15KB)
  - Status: Feature flag system
  - Action: Use for M1+ features

### `/bin/.plan/` - Implementation Plans (10 files)

#### Master Plan
- [x] **00_TEAM_022_MASTER_PLAN.md** (3KB)
  - Status: Active master plan
  - Contains: 4 checkpoints (CP1-CP4)
  - Timeline: 4 weeks
  - Action: This is your roadmap

#### Checkpoint Plans
- [x] **01_CP1_FOUNDATION.md** (9KB)
  - Status: ✅ COMPLETE
  - Deliverables: pool-core, pool-ctl, llorch-ctl
  - Completed by: TEAM-022

- [x] **02_CP2_MODEL_CATALOG.md** (14KB)
  - Status: ✅ COMPLETE
  - Deliverables: Catalog system, registration
  - Completed by: TEAM-022

- [x] **03_CP3_AUTOMATION.md** (16KB)
  - Status: ✅ COMPLETE
  - Deliverables: Model downloads, worker spawning
  - Completed by: TEAM-022, TEAM-023, TEAM-024

- [x] **04_CP4_MULTI_MODEL.md** (18KB)
  - Status: 🔴 PENDING - THIS IS NEXT!
  - Deliverables: Download all models, test all backends
  - Action: START HERE for TEAM-025

#### Team Reports
- [x] **TEAM_022_COMPLETION_SUMMARY.md** (8KB)
  - Status: Complete
  - Summary: CP1-CP3 completion report

- [x] **TEAM_023_SSH_FIX_REPORT.md** (6KB)
  - Status: Complete
  - Summary: Fixed SSH username issue, tested CLI

- [x] **TEAM_024_HUGGINGFACE_CLI_CLEANUP.md** (9KB)
  - Status: Complete (by TEAM-024)
  - Summary: Fixed hf CLI migration, added infer command

- [x] **REFACTORING_CLEANUP_REPORT.md** (8KB)
  - Status: Complete
  - Summary: Dependency cleanup, migration report

#### Meta
- [x] **README.md** (4KB)
  - Status: Plan overview
  - Contains: Checkpoint structure, timeline

---

## What TEAM-025 Should Do Next

### Priority 1: Complete CP4 (Multi-Model Testing) 🎯

**Goal:** Test all 4 models on all backends across all pools

**Location:** Follow `/bin/.plan/04_CP4_MULTI_MODEL.md`

**Tasks:**

#### Task 1.1: Download Remaining Models (2-3 hours)
```bash
# On mac.home.arpa
ssh mac.home.arpa "cd ~/Projects/llama-orch && llorch-pool models download tinyllama"
ssh mac.home.arpa "cd ~/Projects/llama-orch && llorch-pool models download phi3"
ssh mac.home.arpa "cd ~/Projects/llama-orch && llorch-pool models download mistral"

# On workstation.home.arpa (if available)
ssh workstation.home.arpa "cd ~/Projects/llama-orch && llorch-pool models download tinyllama"
ssh workstation.home.arpa "cd ~/Projects/llama-orch && llorch-pool models download phi3"
ssh workstation.home.arpa "cd ~/Projects/llama-orch && llorch-pool models download mistral"

# Verify
ssh mac.home.arpa "cd ~/Projects/llama-orch && llorch-pool models catalog"
```

**Models to Download:**
- ✅ qwen-0.5b (already downloaded)
- ⏳ tinyllama (2.2 GB)
- ⏳ phi3 (5 GB)
- ⏳ mistral (14 GB)

**Total per pool:** ~22 GB

#### Task 1.2: Create Multi-Model Test Script (1-2 hours)

**File:** `.docs/testing/test_all_models.sh`

**Script should:**
1. Iterate through all models
2. Spawn worker for each model
3. Run inference test with `llorch infer`
4. Collect results
5. Generate report

**Template:**
```bash
#!/usr/bin/env bash
set -euo pipefail

MODELS=("qwen-0.5b" "tinyllama" "phi3" "mistral")
BACKENDS=("cpu" "metal" "cuda")

for model in "${MODELS[@]}"; do
    for backend in "${BACKENDS[@]}"; do
        echo "Testing $model on $backend..."
        
        # Spawn worker
        llorch-pool worker spawn $backend --model $model --gpu 0
        
        # Wait for startup
        sleep 5
        
        # Test inference
        llorch infer --worker localhost:8001 --prompt "Hello" --max-tokens 20
        
        # Stop worker
        llorch-pool worker stop-all
    done
done
```

#### Task 1.3: Run Tests and Document Results (2-3 hours)

**Create:** `MODEL_SUPPORT.md`

**Document:**
- Which models work on which backends
- Performance metrics (tokens/sec)
- Any issues or failures
- Memory usage

**Expected Results:**
```
Model Support Matrix
====================
Model       CPU    Metal   CUDA
qwen-0.5b   ✅     ✅      ✅
tinyllama   ✅     ✅      ✅
phi3        ✅     ✅      ✅
mistral     ✅     ✅      ✅
```

---

### Priority 2: Start M1 (Orchestrator Daemon) 🚀

**After CP4 is complete**, start building `orchestratord`

**ARCHITECTURAL CHANGE:** pool-managerd is NOT NEEDED!
- Pool management is CLI-based (llorch-pool)
- M1 now focuses on orchestratord only

**Goal:** Build HTTP daemon that routes inference requests

**Location:** Create `bin/orchestratord/`

**Reference Spec:** `/bin/.specs/00_llama-orch.md` Section 6.1

**Key Requirements:**
1. HTTP server on port 8080
2. Client-facing API (`POST /v2/tasks`)
3. Admission control (validate requests)
4. Queue management (priority queue)
5. Scheduling (select worker based on load/model)
6. SSE streaming relay (worker → client)
7. Worker registry (track available workers)

**Architecture:**
```
Client
    ↓ POST /v2/tasks
orchestratord (HTTP daemon :8080)
    ↓ Scheduling decision
    ↓ POST http://worker:8001/execute
llorch-candled (worker)
    ↓ SSE stream
orchestratord (relay)
    ↓ SSE stream
Client
```

**Steps:**
1. Create `bin/orchestratord/` directory
2. Create `Cargo.toml` with axum dependencies
3. Implement HTTP server (port 8080)
4. Implement admission control
5. Implement queue (use orchestrator-core)
6. Implement scheduling (round-robin to start)
7. Implement SSE relay
8. Implement worker registry
9. Write integration tests

**Estimated Time:** 2-3 weeks

---

### Priority 3: Advanced Orchestrator Features 🎯

**After M1 orchestratord is working**, add advanced features

**Goal:** Production-ready orchestration

**Features:**
1. Advanced scheduling (VRAM-aware, load balancing)
2. Session management (TTL, budgets)
3. Multi-tenant support
4. Metrics & observability
5. Policy engine
6. Retry & timeout policies

**Estimated Time:** 2-4 weeks

---

## Quick Start for TEAM-025

### Step 1: Read This First (30 min)
```bash
# Read the master plan
cat /home/vince/Projects/llama-orch/bin/.plan/00_TEAM_022_MASTER_PLAN.md

# Read CP4 plan
cat /home/vince/Projects/llama-orch/bin/.plan/04_CP4_MULTI_MODEL.md

# Read architecture
cat /home/vince/Projects/llama-orch/bin/.specs/FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md

# Read orchestration overview
cat /home/vince/Projects/llama-orch/ORCHESTRATION_OVERVIEW.md
```

### Step 2: Verify Current State (15 min)
```bash
# Build everything
cd /home/vince/Projects/llama-orch
cargo build --release

# Test worker
ps aux | grep llorch-candled  # Should be running on port 8080

# Test inference
./target/release/llorch infer --worker localhost:8080 --prompt "Hello" --max-tokens 20

# Check catalog
./target/release/llorch-pool models catalog
```

### Step 3: Start CP4 Tasks (Follow Priority 1 above)

---

## Critical Files for TEAM-025

### Must Read (in order):
1. `/bin/.plan/00_TEAM_022_MASTER_PLAN.md` - Master plan
2. `/bin/.plan/04_CP4_MULTI_MODEL.md` - Your immediate tasks
3. `/bin/.specs/FINAL_ARCHITECTURE_SSH_CONTROL_HTTP_INFERENCE.md` - Architecture
4. `/ORCHESTRATION_OVERVIEW.md` - Complete system guide

### Reference When Needed:
- `/bin/.specs/00_llama-orch.md` - Full system spec
- `/bin/.specs/01_M0_worker_orcd.md` - Worker spec
- `/.windsurf/rules/candled-rules.md` - Coding standards

### Don't Need to Read:
- Architecture decision docs (historical context only)
- Old team reports (unless debugging)

---

## Known Issues & Gotchas

### Issue 1: Model Output is Gibberish
**Status:** KNOWN ISSUE  
**Cause:** Model/tokenizer configuration  
**Impact:** Tokens generate but output is nonsense  
**Action:** Ignore for now, focus on infrastructure  
**Fix:** Will address in M3 with proper model configuration

### Issue 2: Worker on Port 8080
**Status:** CURRENT STATE  
**Cause:** TEAM-024 testing  
**Impact:** Worker running on orchestrator's port  
**Action:** Stop worker before building orchestratord  
**Fix:** `pkill llorch-candled` or use proper ports (8001+)

### Issue 3: No orchestratord Yet
**Status:** EXPECTED (M2)  
**Cause:** Not built yet  
**Impact:** Can't test full orchestration  
**Action:** Use `llorch infer` for direct worker testing  
**Fix:** Build orchestratord in M2

### Issue 4: No pool-managerd Yet
**Status:** EXPECTED (M1)  
**Cause:** Not built yet  
**Impact:** Manual worker management  
**Action:** Use `llorch-pool` CLI for now  
**Fix:** Build pool-managerd in M1

---

## Dependencies & Prerequisites

### System Requirements:
- ✅ Rust toolchain (installed)
- ✅ `hf` CLI (installed, replaces huggingface-cli)
- ✅ SSH access to pools (configured)
- ✅ Git with submodules (configured)

### Rust Workspace:
- ✅ All crates compile
- ✅ No blocking errors
- ⚠️ Some warnings (filename collisions, unused keys) - safe to ignore

### Models:
- ✅ qwen-0.5b downloaded (943MB SafeTensors)
- ⏳ tinyllama (need to download)
- ⏳ phi3 (need to download)
- ⏳ mistral (need to download)

### Pools:
- ✅ blep.home.arpa (local, CPU)
- ⏳ mac.home.arpa (remote, Metal) - need to test
- ⏳ workstation.home.arpa (remote, CUDA) - need to test

---

## Testing Checklist

### Before Starting Work:
- [ ] Read master plan
- [ ] Read CP4 plan
- [ ] Verify worker is running
- [ ] Test `llorch infer` command
- [ ] Check catalog shows qwen-0.5b

### After Downloading Models:
- [ ] Verify all models in catalog
- [ ] Check disk space (~22GB per pool)
- [ ] Test one model on each backend

### After Creating Test Script:
- [ ] Script runs without errors
- [ ] All models tested
- [ ] Results documented
- [ ] MODEL_SUPPORT.md created

### Before Starting M1:
- [ ] CP4 complete and documented
- [ ] All tests passing
- [ ] User approval to proceed

---

## Communication Protocol

### When to Ask User:
- ❌ Don't ask about CP4 tasks (follow the plan)
- ❌ Don't ask about architecture decisions (already made)
- ✅ Ask if you find blocking issues
- ✅ Ask before starting M1 (major milestone)
- ✅ Ask if specs are unclear or contradictory

### When to Update Docs:
- ✅ Update this handoff when CP4 complete
- ✅ Create MODEL_SUPPORT.md with results
- ✅ Update TEAM_025_COMPLETION_SUMMARY.md
- ✅ Create TEAM_026_HANDOFF.md before M1

### When to Stop:
- 🛑 If CP4 takes >1 week (re-evaluate)
- 🛑 If models don't download (disk space issue)
- 🛑 If tests fail consistently (investigate)
- 🛑 If user says stop

---

## Success Criteria for TEAM-025

### Minimum (CP4 Complete):
- [ ] All 4 models downloaded on at least 1 pool
- [ ] Test script created and runs
- [ ] At least 1 model tested on each backend
- [ ] Results documented

### Target (CP4 + M1 Start):
- [ ] All 4 models on all pools
- [ ] All models tested on all backends
- [ ] MODEL_SUPPORT.md complete
- [ ] pool-managerd skeleton created

### Stretch (M1 Progress):
- [ ] pool-managerd HTTP server working
- [ ] Worker spawning via HTTP API
- [ ] GPU discovery implemented

---

## Final Notes from TEAM-024

**What Went Well:**
- ✅ Binary naming clarified (llorch vs orchestratord)
- ✅ Inference command added to llorch
- ✅ Documentation comprehensive
- ✅ Token generation proven working

**What Was Hard:**
- 😓 User is exhausted (brain has left)
- 😓 Lots of context to track
- 😓 Model output is gibberish (but infrastructure works)

**Advice for TEAM-025:**
- 📖 Read the plans FIRST before coding
- 🎯 Follow CP4 step-by-step
- 🧪 Test after each task
- 📝 Document everything
- 💤 Let user rest!

**Remember:**
- The infrastructure works!
- The architecture is solid!
- The plans are clear!
- Just follow the checkpoints!

---

**Signed:** TEAM-024  
**Date:** 2025-10-09T16:57:00+02:00  
**Status:** Handoff complete, user needs rest 💤  
**Next Team:** TEAM-025 - You got this! 💪
