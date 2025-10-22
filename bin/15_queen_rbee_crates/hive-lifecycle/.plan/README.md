# Hive Lifecycle Migration Plan

**Created by:** TEAM-208  
**Peer Reviewed by:** TEAM-209  
**Status:** ✅ PLANS REVIEWED - READY FOR IMPLEMENTATION

## ⚠️  TEAM-209 PEER REVIEW COMPLETE

**Date:** 2025-10-22  
**Result:** ✅ **APPROVED WITH UPDATES**  
**Implementation Status:** 0% (no code migrated yet)

### Critical Findings

1. 🔴 **device-detection dependency** missing from plans → FIXED
2. 🟡 **Binary path resolution** inconsistency → DOCUMENTED  
3. 🟢 **LOC count** inaccuracy (1,115 vs 1,114) → CORRECTED

**📖 READ FIRST:** `TEAM_209_CHANGELOG.md` - Complete peer review with all findings and updates

---

## 📋 Quick Start

**New to this project?** → Read `START_HERE.md` first\!

**Looking for your assignment?** → Check the table below

**Want the big picture?** → Read `00_MASTER_PLAN.md`

---

## 👥 Team Assignments

| Team | Phase | Document | Task | LOC | Status |
|------|-------|----------|------|-----|--------|
| TEAM-210 | 1 | `01_PHASE_1_FOUNDATION.md` | Foundation & Types | ~150 | 🔴 NOT STARTED |
| TEAM-211 | 2 | `02_PHASE_2_SIMPLE_OPERATIONS.md` | List/Get/Status | ~100 | 🔴 NOT STARTED |
| TEAM-212 | 3 | `03_PHASE_3_LIFECYCLE_CORE.md` | Start/Stop | ~350 | 🔴 NOT STARTED |
| TEAM-213 | 4 | `04_PHASE_4_INSTALL_UNINSTALL.md` | Install/Uninstall | ~220 | 🔴 NOT STARTED |
| TEAM-214 | 5 | `05_PHASE_5_CAPABILITIES.md` | Capabilities Refresh | ~100 | 🔴 NOT STARTED |
| TEAM-215 | 6 | `06_PHASE_6_INTEGRATION.md` | Wire Everything Up | ~50 | 🔴 NOT STARTED |
| TEAM-209 | 7 | `07_PHASE_7_PEER_REVIEW.md` | Critical Review | N/A | ✅ COMPLETE |

---

## 📊 Migration Overview

### Before (TEAM-209: VERIFIED)
```
job_router.rs: 1,114 LOC (actual count)
├─ Job routing logic (~350 LOC)
└─ Hive operations (~760 LOC) ← MIGRATE THIS
```

### After
```
job_router.rs: ~350 LOC (routing only)
hive-lifecycle crate: ~900 LOC (all hive ops)
```

**Target:** 65% LOC reduction in job_router.rs

---

## 🔄 Workflow

```
TEAM-210 (Foundation)
    ↓
    ├─→ TEAM-211 (Simple Ops) ─┐
    ├─→ TEAM-212 (Lifecycle)   ├─→ TEAM-215 (Integration)
    ├─→ TEAM-213 (Install)     │        ↓
    └─→ TEAM-214 (Capabilities)┘   TEAM-209 (Peer Review)
```

**Parallel Work:** Teams 211-214 can work simultaneously after TEAM-210 completes

---

## 📁 Document Index

### Planning Documents
- **00_MASTER_PLAN.md** - Overall strategy and architecture
- **START_HERE.md** - Entry point for all teams
- **TEAM-208-SUMMARY.md** - Planning phase summary

### Phase Documents (Implementation)
- **01_PHASE_1_FOUNDATION.md** - TEAM-210: Module structure, types, validation
- **02_PHASE_2_SIMPLE_OPERATIONS.md** - TEAM-211: List, Get, Status operations
- **03_PHASE_3_LIFECYCLE_CORE.md** - TEAM-212: Start, Stop operations
- **04_PHASE_4_INSTALL_UNINSTALL.md** - TEAM-213: Install, Uninstall operations
- **05_PHASE_5_CAPABILITIES.md** - TEAM-214: Capabilities refresh
- **06_PHASE_6_INTEGRATION.md** - TEAM-215: Wire up job_router.rs
- **07_PHASE_7_PEER_REVIEW.md** - TEAM-209: Critical review

---

## ⚠️ Critical Requirements

### 1. SSE Routing
**ALL narration MUST include `.job_id(&job_id)`**

```rust
// ❌ WRONG
NARRATE.action("hive_start").human("Starting").emit();

// ✅ CORRECT
NARRATE.action("hive_start").job_id(&job_id).human("Starting").emit();
```

### 2. Error Messages
**Preserve exact error messages from original code**

### 3. Code Signatures
**Add TEAM-XXX signatures to all code**

```rust
// TEAM-210: Created foundation structure
// TEAM-211: Implemented list operation
```

---

## 🧪 Testing Commands

### Build
```bash
cargo check -p queen-rbee-hive-lifecycle
cargo build --bin rbee-keeper --bin queen-rbee --bin rbee-hive
```

### Test Operations
```bash
./rbee hive list
./rbee hive install
./rbee hive start
./rbee hive status
./rbee hive refresh
./rbee hive stop
```

### Verify LOC Reduction
```bash
wc -l bin/10_queen_rbee/src/job_router.rs
# Should be ~350 LOC after migration (was 1,114)
```

---

## 📈 Success Metrics

- ✅ job_router.rs reduced from 1,114 → 350 LOC (65% reduction)
- ✅ All 9 operations migrated to hive-lifecycle crate
- ✅ Clean separation of concerns
- ✅ No regressions in functionality
- ✅ SSE routing works correctly
- ✅ Error messages preserved

---

## 🚀 Getting Started

1. **Find your team number** in the table above
2. **Read `START_HERE.md`** for workflow instructions
3. **Read your phase document** (01-07)
4. **Read engineering rules** at `.windsurf/rules/engineering-rules.md`
5. **Check dependencies** before starting
6. **Implement your phase** following the plan
7. **Verify your work** against acceptance criteria
8. **Update status** in this README

---

## 📞 Need Help?

- **Architecture questions?** → Read `00_MASTER_PLAN.md`
- **Implementation questions?** → Read your phase document
- **Testing questions?** → See "Testing Commands" above
- **Rules questions?** → Read `engineering-rules.md`

---

## ✅ Completion Checklist

- [ ] TEAM-210: Foundation complete
- [ ] TEAM-211: Simple operations complete
- [ ] TEAM-212: Lifecycle core complete
- [ ] TEAM-213: Install/Uninstall complete
- [ ] TEAM-214: Capabilities complete
- [ ] TEAM-215: Integration complete
- [x] TEAM-209: Peer review APPROVED ✅
- [ ] All tests passing
- [ ] LOC reduction verified (65%)
- [ ] No regressions found

---

**Status:** 🔴 NOT STARTED → 🟡 IN PROGRESS → 🟢 COMPLETE

**Next Step:** TEAM-210 starts Phase 1 (Foundation)
