# Visual Guide: rbee-hive Implementation Plan

**Created by:** TEAM-266  
**Date:** Oct 23, 2025

---

## 📂 Document Map

```
bin/.plan/
│
├── 🚀 START_HERE_267_TO_275_RBEE_HIVE_IMPLEMENTATION.md  ← READ THIS FIRST
│
├── 📋 RBEE_HIVE_IMPLEMENTATION_CHECKLIST.md              ← Track your progress
├── 📖 README_RBEE_HIVE_PLAN.md                           ← Quick start guide
├── 📊 VISUAL_GUIDE_RBEE_HIVE_PLAN.md                     ← This file
│
├── 🔍 Investigation (TEAM-266)
│   ├── TEAM_266_MODE_3_INVESTIGATION_FINDINGS.md         ← Full investigation (800+ lines)
│   ├── TEAM_266_HANDOFF.md                               ← Summary (2 pages)
│   ├── TEAM_266_QUICK_REFERENCE.md                       ← TL;DR (1 page)
│   └── TEAM_266_FINAL_SUMMARY.md                         ← What TEAM-266 delivered
│
├── 🏗️ Phase Guides
│   ├── TEAM_267_MODEL_CATALOG_TYPES.md                   ← Phase 1 (20-24h)
│   ├── TEAM_268_MODEL_CATALOG_OPERATIONS.md              ← Phase 2 (16-20h)
│   ├── TEAM_269_TO_272_IMPLEMENTATION_GUIDES.md          ← Phases 3-6 (100-128h)
│   └── TEAM_273_TO_275_FINAL_PHASES.md                   ← Phases 7-9 (62-102h)
│
└── 📁 Archives
    ├── .archive-teams-216-265/                           ← Old team docs
    ├── .archive-phase-guides/                            ← Old phase guides
    ├── .archive-testing-docs/                            ← Testing docs
    ├── .archive-ssh-docs/                                ← SSH docs
    └── .archive-dead-code/                               ← Dead code audits
```

---

## 🗺️ The 9-Phase Journey

```
┌─────────────────────────────────────────────────────────────────────┐
│                     rbee-hive Implementation                        │
│                     198-274 hours (5-7 weeks)                       │
└─────────────────────────────────────────────────────────────────────┘

Phase 1: Model Catalog Types (TEAM-267) ────────────┐
  20-24 hours                                        │
  ├─ ModelEntry struct                              │
  ├─ ModelCatalog with Arc<Mutex<>>                 │
  └─ Unit tests                                      │
                                                     │
Phase 2: Model Catalog Operations (TEAM-268) ───────┤
  16-20 hours                                        │
  ├─ ModelList operation                            │
  ├─ ModelGet operation                             │  MODEL
  └─ ModelDelete operation                          │  MANAGEMENT
                                                     │  COMPLETE
Phase 3: Model Provisioner (TEAM-269) ──────────────┤
  24-32 hours                                        │
  ├─ download_model() function                      │
  ├─ Progress tracking                              │
  └─ ModelDownload operation                        │
                                                     │
                                                     ↓
Phase 4: Worker Registry (TEAM-270) ────────────────┐
  20-24 hours                                        │
  ├─ WorkerEntry struct                             │
  ├─ WorkerRegistry with Arc<Mutex<>>               │
  └─ Unit tests                                      │
                                                     │
Phase 5: Worker Lifecycle - Spawn (TEAM-271) ───────┤
  32-40 hours (MOST COMPLEX)                        │
  ├─ spawn_worker() function                        │  WORKER
  ├─ Port allocation                                │  MANAGEMENT
  ├─ Process spawning                               │  COMPLETE
  └─ WorkerSpawn operation                          │
                                                     │
Phase 6: Worker Lifecycle - Management (TEAM-272) ──┤
  24-32 hours                                        │
  ├─ WorkerList operation                           │
  ├─ WorkerGet operation                            │
  └─ WorkerDelete operation                         │
                                                     │
                                                     ↓
Phase 7: Job Router Integration (TEAM-273) ─────────┐
  16-20 hours                                        │
  ├─ Wire up all 8 operations                       │
  ├─ Remove all TODO markers                        │  HTTP MODE
  └─ Clean compilation                              │  COMPLETE
                                                     │
Phase 8: HTTP Testing & Validation (TEAM-274) ──────┤
  16-24 hours                                        │
  ├─ Test all 8 operations                          │
  ├─ Performance baselines                          │
  └─ Test report                                    │
                                                     │
                                                     ↓
Phase 9: Mode 3 Implementation (TEAM-275) ──────────┐
  30-58 hours                                        │
  ├─ IntegratedHive struct                          │  MODE 3
  ├─ execute_integrated() function                  │  COMPLETE
  ├─ Updated routing                                │
  └─ 110x speedup achieved                          │  🎉
                                                     │
                                                     ↓
                                            ✅ DONE!
```

---

## 🎯 The 8 Operations

```
┌──────────────────────────────────────────────────────────────┐
│                    Worker Operations                         │
├──────────────────────────────────────────────────────────────┤
│ WorkerSpawn   │ Phase 5 │ Spawn worker process              │
│ WorkerList    │ Phase 6 │ List all workers                  │
│ WorkerGet     │ Phase 6 │ Get worker details                │
│ WorkerDelete  │ Phase 6 │ Stop and remove worker            │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                    Model Operations                          │
├──────────────────────────────────────────────────────────────┤
│ ModelDownload │ Phase 3 │ Download model from HuggingFace   │
│ ModelList     │ Phase 2 │ List all models                   │
│ ModelGet      │ Phase 2 │ Get model details                 │
│ ModelDelete   │ Phase 2 │ Remove model from catalog         │
└──────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Architecture Evolution

### Current State (Before Implementation)

```
rbee-keeper (CLI)
      ↓
queen-rbee (Scheduler)
      ↓
hive_forwarder::forward_to_hive()
      ↓
POST http://localhost:8600/v1/jobs
      ↓
rbee-hive (Worker Manager)
      ↓
job_router::route_operation()
      ↓
⚠️  TODO STUBS ⚠️
```

### After Phases 1-7 (HTTP Mode Complete)

```
rbee-keeper (CLI)
      ↓
queen-rbee (Scheduler)
      ↓
hive_forwarder::forward_to_hive()
      ↓
POST http://localhost:8600/v1/jobs
      ↓
rbee-hive (Worker Manager)
      ↓
job_router::route_operation()
      ↓
✅ Real Implementations ✅
      ├─ model-catalog
      ├─ model-provisioner
      └─ worker-lifecycle
```

### After Phase 9 (Mode 3 Complete)

```
rbee-keeper (CLI)
      ↓
queen-rbee (Scheduler)
      ↓
hive_forwarder::forward_to_hive()
      ↓
Mode Detection
      ├─ Remote: HTTP to remote hive
      ├─ Localhost HTTP: POST http://localhost:8600/v1/jobs
      └─ Integrated: ⚡ Direct function calls (110x faster!)
            ↓
      execute_integrated()
            ↓
      ✅ In-process calls ✅
            ├─ model-catalog
            ├─ model-provisioner
            └─ worker-lifecycle
```

---

## 📊 Effort Distribution

```
Model Management (Phases 1-3): 60-76 hours
├─ Phase 1: 20-24h ████████
├─ Phase 2: 16-20h ██████
└─ Phase 3: 24-32h ██████████

Worker Management (Phases 4-6): 76-96 hours
├─ Phase 4: 20-24h ████████
├─ Phase 5: 32-40h ████████████████ ← LONGEST
└─ Phase 6: 24-32h ██████████

Integration & Testing (Phases 7-8): 32-44 hours
├─ Phase 7: 16-20h ██████
└─ Phase 8: 16-24h ████████

Mode 3 (Phase 9): 30-58 hours
└─ Phase 9: 30-58h ████████████████████

Total: 198-274 hours ████████████████████████████████████████
```

---

## 🔄 Workflow Per Phase

```
┌─────────────────────────────────────────────────────────┐
│  1. Read your guide (TEAM_XXX_*.md)                     │
│  2. Read previous handoff (TEAM_YYY_HANDOFF.md)         │
│  3. Verify previous work (cargo check)                  │
│  4. Implement your phase                                │
│  5. Add narration events                                │
│  6. Write unit tests                                    │
│  7. Verify compilation (cargo check + cargo test)       │
│  8. Create handoff (TEAM_XXX_HANDOFF.md)                │
│  9. Mark complete in checklist                          │
└─────────────────────────────────────────────────────────┘
```

---

## 🎓 Key Patterns

### 1. Narration Pattern

```rust
NARRATE
    .action("operation_start")
    .job_id(&job_id)  // ← CRITICAL for SSE routing!
    .context(&value)
    .human("Doing something with {}")
    .emit();
```

### 2. State Management Pattern

```rust
pub struct Registry {
    items: Arc<Mutex<HashMap<String, Item>>>,
}
```

### 3. Error Handling Pattern

```rust
match operation() {
    Ok(result) => {
        NARRATE.action("success").job_id(&job_id).emit();
    }
    Err(e) => {
        NARRATE.action("error").job_id(&job_id).emit();
        return Err(e);
    }
}
```

---

## 🚨 Critical Dependencies

```
Phase 1 ──→ Phase 2 ──→ Phase 3
                              ↓
Phase 4 ──→ Phase 5 ──→ Phase 6
                              ↓
                        Phase 7 ──→ Phase 8 ──→ Phase 9
```

**You cannot skip phases!** Each phase depends on the previous one.

---

## 📈 Progress Tracking

### Use the Checklist

```bash
# Open the checklist
cat RBEE_HIVE_IMPLEMENTATION_CHECKLIST.md

# Check progress
grep -c "✅" RBEE_HIVE_IMPLEMENTATION_CHECKLIST.md

# See what's left
grep "⬜ TODO" RBEE_HIVE_IMPLEMENTATION_CHECKLIST.md
```

### Quick Status

```
Phase 1: [ ] Model Catalog Types
Phase 2: [ ] Model Catalog Operations
Phase 3: [ ] Model Provisioner
Phase 4: [ ] Worker Registry
Phase 5: [ ] Worker Lifecycle - Spawn
Phase 6: [ ] Worker Lifecycle - Management
Phase 7: [ ] Job Router Integration
Phase 8: [ ] HTTP Testing
Phase 9: [ ] Mode 3 Implementation
```

---

## 🎉 Success Metrics

### After Phase 7
- ✅ All 8 operations work via HTTP
- ✅ `cargo build --bin rbee-hive` succeeds
- ✅ No TODO markers in job_router.rs

### After Phase 8
- ✅ Integration tests passing
- ✅ Performance: ~1.1ms per operation (HTTP)
- ✅ Known limitations documented

### After Phase 9
- ✅ Mode 3 working for localhost
- ✅ Performance: ~0.01ms per operation (Integrated)
- ✅ Speedup: 110x measured
- ✅ No breaking changes

---

## 📞 Need Help?

### Read These In Order

1. `START_HERE_267_TO_275_RBEE_HIVE_IMPLEMENTATION.md` - Master plan
2. `README_RBEE_HIVE_PLAN.md` - Quick start
3. Your phase guide (`TEAM_XXX_*.md`)
4. `TEAM_266_MODE_3_INVESTIGATION_FINDINGS.md` - Context
5. `RBEE_HIVE_IMPLEMENTATION_CHECKLIST.md` - Track progress

### Still Stuck?

Document your question in your handoff for the next team.

---

**TEAM-266 signing off. You've got this! 🐝**
