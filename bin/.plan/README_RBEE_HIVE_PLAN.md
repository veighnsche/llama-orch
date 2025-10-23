# rbee-hive Implementation Plan - README

**Created by:** TEAM-266  
**Date:** Oct 23, 2025  
**Status:** 🔴 READY TO START

---

## 🎯 What Is This?

This is a **9-phase implementation plan** to build out all rbee-hive crates and enable Mode 3 (Integrated) communication between queen-rbee and rbee-hive.

**Goal:** Implement worker and model management functionality, then achieve 110x performance improvement for localhost operations.

---

## 📚 Documents Created

### Start Here
1. **START_HERE_267_TO_275_RBEE_HIVE_IMPLEMENTATION.md** - Master plan overview
2. **RBEE_HIVE_IMPLEMENTATION_CHECKLIST.md** - Complete checklist for all phases

### Phase Guides
3. **TEAM_267_MODEL_CATALOG_TYPES.md** - Phase 1: Model catalog types
4. **TEAM_268_MODEL_CATALOG_OPERATIONS.md** - Phase 2: Model operations
5. **TEAM_269_TO_272_IMPLEMENTATION_GUIDES.md** - Phases 3-6: Model provisioner & worker lifecycle
6. **TEAM_273_TO_275_FINAL_PHASES.md** - Phases 7-9: Integration, testing, Mode 3

### Context
7. **STORAGE_ARCHITECTURE.md** - Filesystem-based storage design ← **IMPORTANT!**
8. **TEAM_266_MODE_3_INVESTIGATION_FINDINGS.md** - Comprehensive investigation (800+ lines)
9. **TEAM_266_HANDOFF.md** - Quick summary (2 pages)
10. **TEAM_266_QUICK_REFERENCE.md** - TL;DR (1 page)

---

## 🚀 Quick Start

### For TEAM-267 (First Team)

```bash
# 1. Read the storage architecture (CRITICAL!)
cat STORAGE_ARCHITECTURE.md

# 2. Read the master plan
cat START_HERE_267_TO_275_RBEE_HIVE_IMPLEMENTATION.md

# 3. Read your phase guide
cat TEAM_267_MODEL_CATALOG_TYPES.md

# 4. Read the investigation for context
cat TEAM_266_MODE_3_INVESTIGATION_FINDINGS.md

# 5. Start implementing!
cd bin/25_rbee_hive_crates/model-catalog
```

### For Subsequent Teams

```bash
# 1. Read your phase guide
cat TEAM_XXX_*.md

# 2. Read previous team's handoff
cat TEAM_YYY_HANDOFF.md

# 3. Check the master checklist
cat RBEE_HIVE_IMPLEMENTATION_CHECKLIST.md

# 4. Start implementing!
```

---

## 📋 The 9 Phases

| Phase | Team | Focus | Effort | Files |
|-------|------|-------|--------|-------|
| 1 | 267 | Model Catalog Types | 20-24h | model-catalog/src/types.rs, catalog.rs |
| 2 | 268 | Model Catalog Operations | 16-20h | rbee-hive/src/job_router.rs |
| 3 | 269 | Model Provisioner | 24-32h | model-provisioner/src/lib.rs |
| 4 | 270 | Worker Registry | 20-24h | worker-lifecycle/src/registry.rs |
| 5 | 271 | Worker Spawn | 32-40h | worker-lifecycle/src/spawn.rs |
| 6 | 272 | Worker Management | 24-32h | rbee-hive/src/job_router.rs |
| 7 | 273 | Job Router Integration | 16-20h | rbee-hive/src/main.rs, job_router.rs |
| 8 | 274 | HTTP Testing | 16-24h | Test all operations |
| 9 | 275 | Mode 3 Implementation | 30-58h | queen-rbee/src/integrated_hive.rs |

**Total:** 198-274 hours (5-7 weeks)

---

## 🎯 Success Criteria

### After Phase 7
- ✅ All 8 operations work via HTTP
- ✅ WorkerSpawn creates processes
- ✅ ModelDownload fetches models
- ✅ All operations emit narration

### After Phase 8
- ✅ Integration tests passing
- ✅ Performance baselines established
- ✅ Known limitations documented

### After Phase 9
- ✅ Mode 3 working for localhost
- ✅ 110x speedup measured
- ✅ No breaking changes

---

## 📁 File Organization

### Implementation Files
```
bin/25_rbee_hive_crates/
├── model-catalog/          ← Phases 1-2
│   ├── src/
│   │   ├── types.rs
│   │   └── catalog.rs
│   └── Cargo.toml
├── model-provisioner/      ← Phase 3
│   └── src/lib.rs
└── worker-lifecycle/       ← Phases 4-6
    └── src/
        ├── registry.rs
        └── spawn.rs

bin/20_rbee_hive/           ← Phases 2-3, 5-7
├── src/
│   ├── main.rs
│   └── job_router.rs
└── Cargo.toml

bin/10_queen_rbee/          ← Phase 9
├── src/
│   ├── integrated_hive.rs
│   └── hive_forwarder.rs
└── Cargo.toml
```

### Documentation Files (in bin/.plan/)
```
START_HERE_267_TO_275_RBEE_HIVE_IMPLEMENTATION.md  ← Read first
RBEE_HIVE_IMPLEMENTATION_CHECKLIST.md              ← Track progress
TEAM_267_MODEL_CATALOG_TYPES.md                    ← Phase 1
TEAM_268_MODEL_CATALOG_OPERATIONS.md               ← Phase 2
TEAM_269_TO_272_IMPLEMENTATION_GUIDES.md           ← Phases 3-6
TEAM_273_TO_275_FINAL_PHASES.md                    ← Phases 7-9
TEAM_266_MODE_3_INVESTIGATION_FINDINGS.md          ← Context
TEAM_266_HANDOFF.md                                ← Summary
TEAM_266_QUICK_REFERENCE.md                        ← TL;DR
README_RBEE_HIVE_PLAN.md                           ← This file
```

---

## 🔄 Workflow

### For Each Phase

1. **Read your guide** - TEAM_XXX_*.md
2. **Check previous work** - Read TEAM_YYY_HANDOFF.md
3. **Verify compilation** - `cargo check`
4. **Implement your phase** - Follow the guide
5. **Add narration** - All operations need events
6. **Write tests** - Unit tests for your code
7. **Verify again** - `cargo check` and `cargo test`
8. **Create handoff** - TEAM_XXX_HANDOFF.md
9. **Mark complete** - Update checklist

### Testing Commands

```bash
# Check a specific crate
cargo check --package rbee-hive-model-catalog

# Test a specific crate
cargo test --package rbee-hive-model-catalog

# Check rbee-hive binary
cargo check --bin rbee-hive

# Build rbee-hive
cargo build --bin rbee-hive

# Run rbee-hive
cargo run --bin rbee-hive -- --port 8600
```

---

## ⚠️ Known Issues

### Pre-existing Blockers

1. **queen-rbee-worker-registry compilation error**
   - Missing `HiveRegistry` type
   - Does not block rbee-hive work
   - Focus on rbee-hive crates first

2. **Worker binary not available**
   - WorkerSpawn will fail without actual worker binary
   - Document this limitation
   - Use placeholder for now

### Expected Limitations

1. **Model download stub** - Phase 3 won't actually download from HuggingFace yet
2. **Process cleanup incomplete** - Phase 6 may not kill processes properly
3. **File deletion incomplete** - ModelDelete may not remove files from disk

**These are acceptable for v0.1.0 - document them and move forward!**

---

## 📊 Progress Tracking

Use the checklist: `RBEE_HIVE_IMPLEMENTATION_CHECKLIST.md`

### Quick Status Check

```bash
# Count completed phases
grep -c "✅" RBEE_HIVE_IMPLEMENTATION_CHECKLIST.md

# See what's left
grep "⬜ TODO" RBEE_HIVE_IMPLEMENTATION_CHECKLIST.md
```

---

## 🎓 Key Concepts

### 1. Narration Pattern

All operations MUST emit narration events with job_id:

```rust
NARRATE
    .action("operation_start")
    .job_id(&job_id)  // ← CRITICAL!
    .context(&some_value)
    .human("Doing something with {}")
    .emit();
```

### 2. State Management

All state uses Arc<Mutex<>> for thread safety:

```rust
pub struct Registry {
    items: Arc<Mutex<HashMap<String, Item>>>,
}
```

### 3. Error Handling

Errors become narration events:

```rust
match do_something() {
    Ok(result) => { /* emit success */ }
    Err(e) => {
        NARRATE
            .action("operation_error")
            .job_id(&job_id)
            .human("❌ Failed: {}")
            .emit();
        return Err(e);
    }
}
```

---

## 🚨 Critical Path

```
Phase 1 (Model Types)
  ↓
Phase 2 (Model Ops)
  ↓
Phase 3 (Model Download) ──┐
  ↓                         │
Phase 4 (Worker Registry)   │
  ↓                         │
Phase 5 (Worker Spawn) ─────┤
  ↓                         │
Phase 6 (Worker Mgmt)       │
  ↓                         │
Phase 7 (Integration) ←─────┘
  ↓
Phase 8 (Testing)
  ↓
Phase 9 (Mode 3)
```

**Phases must be completed in order!**

---

## 📞 Questions?

1. Read `START_HERE_267_TO_275_RBEE_HIVE_IMPLEMENTATION.md`
2. Read your phase guide
3. Read `TEAM_266_MODE_3_INVESTIGATION_FINDINGS.md`
4. Check the checklist
5. Read previous team's handoff

Still stuck? Document your question in your handoff for the next team.

---

## 🎉 Final Goal

After all 9 phases:

- ✅ All 8 operations working via HTTP
- ✅ Mode 3 working for localhost
- ✅ 110x speedup for list/get operations
- ✅ Full test coverage
- ✅ Complete documentation

**Let's build this! 🐝**

---

**TEAM-266 signing off. Good luck, teams 267-275!**
