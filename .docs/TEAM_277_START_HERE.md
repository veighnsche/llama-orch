# TEAM-277+ Declarative Lifecycle Migration - START HERE

**Mission:** Migrate from imperative to declarative lifecycle management  
**Total Effort:** 64-90 hours  
**Teams:** 6 teams (TEAM-278 through TEAM-283)  
**Date:** Oct 23, 2025

---

## 🎯 Mission Overview

Transform rbee from **imperative operations** (manual commands) to **declarative configuration** (config-driven sync).

**Before:**
```bash
rbee install-hive --alias gpu-1
rbee install-worker --hive gpu-1 --type vllm
# Sequential, slow, error-prone
```

**After:**
```bash
cat > ~/.config/rbee/hives.conf << 'EOF'
[[hive]]
alias = "gpu-1"
workers = [{ type = "vllm" }]
EOF

rbee sync  # ✅ Installs everything concurrently (3-10x faster!)
```

---

## 📚 Required Reading (ALL TEAMS)

**Before starting your phase, read these:**

### Architecture & Design
1. `.docs/DECLARATIVE_CONFIG_ANALYSIS.md` - Why declarative?
2. `.docs/DECLARATIVE_MIGRATION_PLAN.md` - Full migration strategy
3. `.docs/PACKAGE_MANAGER_OPERATIONS.md` - New operations design
4. `.arch/00_OVERVIEW_PART_1.md` - System architecture
5. `.arch/01_COMPONENTS_PART_2.md` - Component responsibilities

### Implementation Guides
1. `bin/ADDING_NEW_OPERATIONS.md` - 3-file pattern for operations
2. `bin/.plan/TEAM_258_CONSOLIDATION_SUMMARY.md` - Operation routing
3. `bin/.plan/TEAM_259_JOB_CLIENT_CONSOLIDATION.md` - JobClient usage

### Your Instructions
1. `.docs/TEAM_277_MASTER_INDEX.md` - Overview and navigation
2. `.docs/TEAM_277_INSTRUCTIONS_PART_1.md` - Overview & Phase 1
3. `.docs/TEAM_277_INSTRUCTIONS_PART_2.md` - Phase 1 & 2 details
4. `.docs/TEAM_277_INSTRUCTIONS_PART_3.md` - Phase 3 details
5. `.docs/TEAM_277_INSTRUCTIONS_PART_4.md` - Phase 4, 5, 6 details
6. `.docs/TEAM_277_CHECKLIST.md` - Complete task checklist
7. `.docs/TEAM_277_CORRECTIONS_APPLIED.md` - Important corrections

---

## 👥 Team Assignments

### TEAM-278: Config Support (8-12 hours)
**Phase:** 1  
**Goal:** Add `hives.conf` parsing and REPLACE old config patterns

**Your Tasks:**
- Create `declarative.rs` with config structs
- Implement TOML parsing
- Add config validation
- Write tests

**Deliverables:**
- `bin/99_shared_crates/rbee-config/src/declarative.rs` (NEW)
- Updated `lib.rs` and `Cargo.toml`
- Working config parser with tests

**Success Criteria:**
- ✅ `cargo check -p rbee-config` passes
- ✅ `cargo test -p rbee-config` passes
- ✅ Config loads from `~/.config/rbee/hives.conf`

**Instructions:** See `TEAM_277_INSTRUCTIONS_PART_2.md` (Phase 1 detailed steps)

**Checklist:** Lines 1-71 in `TEAM_277_CHECKLIST.md`

---

### TEAM-279: Package Operations (12-16 hours)
**Phase:** 2  
**Goal:** Add new operations to Operation enum

**Your Tasks:**
- Add 6 new package operations to enum
- Update `Operation::name()` method
- Update `should_forward_to_hive()` docs
- Verify compilation

**Deliverables:**
- Updated `bin/99_shared_crates/rbee-operations/src/lib.rs`
- All 6 operations compile

**Success Criteria:**
- ✅ `cargo check -p rbee-operations` passes
- ✅ `cargo test -p rbee-operations` passes
- ✅ All operations have correct names

**Instructions:** See `TEAM_277_INSTRUCTIONS_PART_2.md` (Phase 2 detailed steps)

**Checklist:** Lines 73-107 in `TEAM_277_CHECKLIST.md`

**Dependencies:** Wait for TEAM-278 to complete Phase 1

---

### TEAM-280: Package Manager Core (24-32 hours)
**Phase:** 3  
**Goal:** Implement package manager logic in queen-rbee

**Your Tasks:**
- Create package_manager module structure
- Implement install.rs (hive + worker installation via SSH)
- Implement sync.rs (orchestration with concurrency)
- Implement diff.rs (state comparison)
- Wire into job_router.rs

**Deliverables:**
- `bin/10_queen_rbee/src/package_manager/` (NEW directory)
- 6 new modules: mod.rs, sync.rs, diff.rs, install.rs, status.rs, validate.rs, migrate.rs
- Updated job_router.rs with package operations

**Success Criteria:**
- ✅ `cargo check -p queen-rbee` passes
- ✅ Sync works end-to-end
- ✅ Concurrent installation works

**Critical Notes:**
- Use `tokio::spawn` for concurrency
- Use `SshClient` from hive-lifecycle for SSH
- Add `.job_id()` to all narration for SSE routing
- Queen installs BOTH hive AND workers via SSH

**Instructions:** See `TEAM_277_INSTRUCTIONS_PART_3.md` (Phase 3 detailed steps)

**Checklist:** Lines 109-186 in `TEAM_277_CHECKLIST.md`

**Dependencies:** Wait for TEAM-279 to complete Phase 2

---

### TEAM-281: Simplify Hive (8-12 hours)
**Phase:** 4  
**Goal:** Remove worker installation logic from rbee-hive

**Your Tasks:**
- Update worker-lifecycle documentation (stubs stay!)
- Remove worker install operations from hive job_router.rs
- Make worker catalog read-only
- Verify compilation

**Deliverables:**
- Updated `bin/25_rbee_hive_crates/worker-lifecycle/src/lib.rs`
- Updated `bin/20_rbee_hive/src/job_router.rs`
- Updated `bin/20_rbee_hive/src/worker_catalog.rs`

**Success Criteria:**
- ✅ `cargo check -p rbee-hive` passes
- ✅ `cargo check -p worker-lifecycle` passes
- ✅ Hive only manages processes, not installation

**Critical Notes:**
- **DO NOT DELETE** `install.rs` and `uninstall.rs` - they are API stubs!
- Update documentation to clarify queen handles installation
- Remove match arms for WorkerDownload/Build/BinaryDelete

**Instructions:** See `TEAM_277_INSTRUCTIONS_PART_4.md` (Phase 4 detailed steps)

**Checklist:** Lines 188-217 in `TEAM_277_CHECKLIST.md`

**Dependencies:** Wait for TEAM-280 to complete Phase 3

---

### TEAM-282: CLI Updates (8-12 hours)
**Phase:** 5  
**Goal:** Add package manager commands to rbee-keeper CLI

**Your Tasks:**
- Create command handler files (sync, status, validate, migrate)
- Update CLI enum with new commands
- Update main.rs with match arms
- Test commands

**Deliverables:**
- `bin/00_rbee_keeper/src/handlers/sync.rs` (NEW)
- `bin/00_rbee_keeper/src/handlers/package_status.rs` (NEW)
- `bin/00_rbee_keeper/src/handlers/validate.rs` (NEW)
- `bin/00_rbee_keeper/src/handlers/migrate.rs` (NEW)
- Updated CLI enum and main.rs

**Success Criteria:**
- ✅ `cargo check -p rbee-keeper` passes
- ✅ `cargo build -p rbee-keeper` succeeds
- ✅ `rbee sync --dry-run` works
- ✅ `rbee status` works

**Critical Notes:**
- Use `job_client::submit_and_stream_job()` NOT `QueenClient`
- Load config with `Config::load()`
- Follow existing command patterns in `handlers/` directory

**Instructions:** See `TEAM_277_INSTRUCTIONS_PART_4.md` (Phase 5 detailed steps)

**Checklist:** Lines 219-252 in `TEAM_277_CHECKLIST.md`

**Dependencies:** Wait for TEAM-281 to complete Phase 4

---

### TEAM-283: Cleanup & Verification (4-6 hours)
**Phase:** 6  
**Goal:** Remove old operations and verify everything works

**Your Tasks:**
- Remove deprecated operations from Operation enum
- Remove old CLI commands
- Update documentation
- Run end-to-end tests
- Write handoff document

**Deliverables:**
- Updated `bin/99_shared_crates/rbee-operations/src/lib.rs`
- Updated `bin/00_rbee_keeper/src/cli/mod.rs`
- Updated `bin/ADDING_NEW_OPERATIONS.md`
- `bin/.plan/TEAM_277_HANDOFF.md` (NEW)

**Success Criteria:**
- ✅ `cargo check --workspace` passes
- ✅ `cargo test --workspace` passes
- ✅ End-to-end test works (see checklist)
- ✅ Old operations removed
- ✅ Handoff document complete (max 2 pages)

**End-to-End Test:**
```bash
# 1. Create config
cat > ~/.config/rbee/hives.conf << 'EOF'
[[hive]]
alias = "test-hive"
hostname = "localhost"
ssh_user = "vince"
workers = [{ type = "vllm", version = "latest" }]
EOF

# 2. Validate
rbee validate

# 3. Dry run
rbee sync --dry-run

# 4. Install
rbee sync

# 5. Check status
rbee status
```

**Instructions:** See `TEAM_277_INSTRUCTIONS_PART_4.md` (Phase 6 detailed steps)

**Checklist:** Lines 254-304 in `TEAM_277_CHECKLIST.md`

**Dependencies:** Wait for TEAM-282 to complete Phase 5

---

## 📊 Progress Tracking

### Overall Status

| Team | Phase | Status | Duration | Completion |
|------|-------|--------|----------|------------|
| TEAM-278 | Phase 1: Config | ✅ DONE | 8h / 8-12h | 100% |
| TEAM-279 | Phase 2: Operations | ⏳ TODO | 0h / 12-16h | 0% |
| TEAM-280 | Phase 3: Package Manager | ⏳ TODO | 0h / 24-32h | 0% |
| TEAM-281 | Phase 4: Simplify Hive | ⏳ TODO | 0h / 8-12h | 0% |
| TEAM-282 | Phase 5: CLI | ⏳ TODO | 0h / 8-12h | 0% |
| TEAM-283 | Phase 6: Cleanup | ⏳ TODO | 0h / 4-6h | 0% |
| **TOTAL** | **All Phases** | **🔄 IN PROGRESS** | **8h / 64-90h** | **12.5%** |

### Update Instructions

Each team should update this table when:
- Starting work: Change status to 🔄 IN PROGRESS
- Making progress: Update duration
- Completing work: Change status to ✅ DONE, update completion to 100%

---

## 🔄 Workflow

### For Each Team

1. **Read your phase instructions** in the TEAM_277_INSTRUCTIONS_PART_X.md files
2. **Check your checklist section** in TEAM_277_CHECKLIST.md
3. **Wait for dependencies** (previous team must complete)
4. **Update status** to 🔄 IN PROGRESS in this file
5. **Complete your tasks** following the checklist
6. **Verify your work** with the success criteria
7. **Update status** to ✅ DONE in this file
8. **Hand off to next team** by notifying them

### Handoff Pattern

When you complete your phase:
1. ✅ Verify all success criteria met
2. ✅ Update progress table above
3. ✅ Check off all items in your checklist section
4. ✅ Commit your changes
5. ✅ Notify next team they can start

---

## 🚨 Critical Rules

### v0.1.0 = BREAK THINGS!

**This is v0.1.0 - breaking changes are EXPECTED and GOOD.**

- ✅ Delete old code aggressively
- ✅ Remove backwards compatibility
- ✅ No shims, no compatibility layers
- ✅ Clean architecture over preserving old patterns
- ❌ Don't be "careful" - be BOLD
- ❌ Don't preserve old operations "just in case"

### From Engineering Rules

1. **Add TEAM-XXX signatures** to all code you write
2. **NO TODO markers** - implement functions or ask for help
3. **NO background testing** - always run tests in foreground
4. **Complete previous team's work** before adding new priorities
5. **Handoffs ≤2 pages** with code examples and progress
6. **v0.1.0 = DESTRUCTIVE IS ALLOWED** - Clean up aggressively

### Project-Specific Rules

1. **Use existing patterns:**
   - `SshClient` from hive-lifecycle for SSH
   - `submit_and_stream_job()` from job_client
   - `tokio::spawn` for concurrency
   - `.job_id()` for all narration (SSE routing)

2. **Don't delete stub files:**
   - `worker-lifecycle/src/install.rs` is a stub - keep it!
   - `worker-lifecycle/src/uninstall.rs` is a stub - keep it!

3. **Config loading pattern:**
   ```rust
   let config = if let Some(path) = config_path {
       HivesConfig::load_from(&path)?
   } else {
       HivesConfig::load()?
   };
   ```

4. **Job submission pattern:**
   ```rust
   let config = Config::load()?;
   let queen_url = config.queen_url();
   submit_and_stream_job(&queen_url, operation).await?;
   ```

---

## 📁 Key Files Reference

### Files to Create (by team)

**TEAM-278:**
- `bin/99_shared_crates/rbee-config/src/declarative.rs`

**TEAM-280:**
- `bin/10_queen_rbee/src/package_manager/mod.rs`
- `bin/10_queen_rbee/src/package_manager/sync.rs`
- `bin/10_queen_rbee/src/package_manager/diff.rs`
- `bin/10_queen_rbee/src/package_manager/install.rs`
- `bin/10_queen_rbee/src/package_manager/status.rs`
- `bin/10_queen_rbee/src/package_manager/validate.rs`
- `bin/10_queen_rbee/src/package_manager/migrate.rs`

**TEAM-282:**
- `bin/00_rbee_keeper/src/handlers/sync.rs`
- `bin/00_rbee_keeper/src/handlers/package_status.rs`
- `bin/00_rbee_keeper/src/handlers/validate.rs`
- `bin/00_rbee_keeper/src/handlers/migrate.rs`

**TEAM-283:**
- `bin/.plan/TEAM_277_HANDOFF.md`

### Files to Modify (by team)

**TEAM-278:**
- `bin/99_shared_crates/rbee-config/src/lib.rs`
- `bin/99_shared_crates/rbee-config/Cargo.toml`

**TEAM-279:**
- `bin/99_shared_crates/rbee-operations/src/lib.rs`

**TEAM-280:**
- `bin/10_queen_rbee/src/job_router.rs`
- `bin/10_queen_rbee/src/lib.rs`

**TEAM-281:**
- `bin/25_rbee_hive_crates/worker-lifecycle/src/lib.rs`
- `bin/20_rbee_hive/src/job_router.rs`
- `bin/20_rbee_hive/src/worker_catalog.rs`

**TEAM-282:**
- `bin/00_rbee_keeper/src/cli/mod.rs`
- `bin/00_rbee_keeper/src/main.rs`

**TEAM-283:**
- `bin/99_shared_crates/rbee-operations/src/lib.rs`
- `bin/00_rbee_keeper/src/cli/mod.rs`
- `bin/ADDING_NEW_OPERATIONS.md`

---

## 🆘 Getting Help

### If You're Stuck

1. **Read the corrections document:** `.docs/TEAM_277_CORRECTIONS_APPLIED.md`
2. **Check existing code patterns** in the codebase
3. **Review previous team's work** for context
4. **Ask specific questions** with file paths and line numbers

### Common Issues

**Issue:** Config won't load  
**Solution:** Check path is `~/.config/rbee/hives.conf` and file exists

**Issue:** SSH connection fails  
**Solution:** Use `SshClient` from `hive-lifecycle/src/ssh_helper.rs`

**Issue:** Narration not showing in SSE  
**Solution:** Add `.job_id(&job_id)` to all narration events

**Issue:** Tests hanging  
**Solution:** Never use background jobs (`&`), always foreground

---

## 🎯 Success Metrics

### When ALL Teams Complete

- ✅ Config-driven lifecycle management works
- ✅ `rbee sync` installs hives + workers concurrently
- ✅ 3-10x faster than sequential installation
- ✅ Config file is source of truth
- ✅ Old imperative operations removed
- ✅ All tests pass
- ✅ Documentation complete

### Expected Results

- **LOC Added:** ~1,500-2,000 lines
- **LOC Removed:** ~500-800 lines
- **Net Change:** ~1,000 lines
- **Performance:** 3-10x faster installation
- **Architecture:** Simpler, more maintainable

---

## 📝 Final Notes

**This is a major architectural improvement.** Take your time, follow the patterns, test thoroughly, and document everything.

**Communication is key.** Update the progress table, notify the next team when you're done, and ask questions if stuck.

**Quality over speed.** It's better to take the full time estimate and deliver solid work than to rush and create bugs.

**Good luck, teams! 🚀**

---

**Last Updated:** Oct 23, 2025  
**Status:** Phase 1 complete (TEAM-278) - Ready for TEAM-279 to start Phase 2
