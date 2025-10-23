# TEAM-277 Instructions - Part 1: Overview & Phase 1

**Mission:** Migrate to declarative lifecycle management  
**Your Job:** Read all docs, understand the plan, execute phases 1-6

---

## 📚 Required Reading

**Read these in order before coding:**

### Architecture (Understand the System)
- `.arch/00_OVERVIEW_PART_1.md`
- `.arch/01_COMPONENTS_PART_2.md`
- `.arch/03_DATA_FLOW_PART_4.md`

### Design (Understand the Plan)
- `.docs/DECLARATIVE_CONFIG_ANALYSIS.md` - Why declarative?
- `.docs/DECLARATIVE_MIGRATION_PLAN.md` - **YOUR ROADMAP**
- `.docs/PACKAGE_MANAGER_OPERATIONS.md` - New operations

### Implementation
- `bin/ADDING_NEW_OPERATIONS.md` - 3-file pattern
- `bin/.plan/TEAM_258_CONSOLIDATION_SUMMARY.md` - Operation routing

### Code to Explore
- `bin/05_rbee_keeper_crates/queen-lifecycle/` - Queen lifecycle
- `bin/15_queen_rbee_crates/hive-lifecycle/` - Hive lifecycle
- `bin/25_rbee_hive_crates/worker-lifecycle/` - Worker lifecycle
- `bin/99_shared_crates/daemon-lifecycle/` - Shared utilities

---

## 🗺️ The 6 Phases

1. **Phase 1:** Config support (8-12h) - Add `hives.conf` parsing
2. **Phase 2:** Operations (12-16h) - Add `PackageSync` etc to Operation enum
3. **Phase 3:** Package manager (24-32h) - Implement sync logic in queen
4. **Phase 4:** Simplify hive (8-12h) - Remove worker install from hive
5. **Phase 5:** CLI (8-12h) - Add `rbee sync` command
6. **Phase 6:** Cleanup (4-6h) - Remove old operations

**Total: 64-90 hours**

---

## Phase 1: Add Config Support (8-12 hours)

### Goal
Add `hives.conf` parsing and REPLACE old config patterns.

**v0.1.0 = BREAK THINGS!** Delete old code, no backwards compatibility needed.

### What to Build
1. Config schema (`HivesConfig`, `HiveConfig`, `WorkerConfig`)
2. TOML parser
3. Config validator

### Files to Create
- `bin/99_shared_crates/rbee-config/src/declarative.rs` (NEW)

### Files to Modify
- `bin/99_shared_crates/rbee-config/src/lib.rs`
- `bin/99_shared_crates/rbee-config/Cargo.toml`

### Step-by-Step

**See TEAM_277_INSTRUCTIONS_PART_2.md for detailed Phase 1 steps**

### Verification
```bash
cargo check -p rbee-config
cargo test -p rbee-config
```

✅ **Phase 1 complete when config parsing works**

---

**Continue to Part 2 for Phase 1 detailed steps**
