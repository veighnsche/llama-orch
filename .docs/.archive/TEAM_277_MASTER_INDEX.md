# TEAM-277: Declarative Lifecycle Migration - Master Index

**Date:** Oct 23, 2025  
**Mission:** Migrate from imperative to declarative lifecycle management  
**Estimated Effort:** 64-90 hours (2-3 weeks)

## 🔥 v0.1.0 = BREAK EVERYTHING!

**This is v0.1.0 - breaking changes are EXPECTED and GOOD.**

- ✅ Delete old code aggressively - no dangling files
- ✅ Remove backwards compatibility - clean architecture
- ✅ No shims or compatibility layers - pure implementation
- ❌ Don't be "careful" - be BOLD and DESTRUCTIVE
- ❌ Don't preserve old operations "just in case"

**From engineering-rules.md:** "v0.1.0 = DESTRUCTIVE IS ALLOWED. Clean up aggressively. No dangling files, no dead code."

---

## 📖 How to Use This Guide

**You are TEAM-277.** Your job is to implement declarative lifecycle management for rbee.

**Read documents in this order:**

### 1. Start Here (This Document)
- Overview of the mission
- Document index
- Quick reference

### 2. Read Instructions (In Order)
- **Part 1:** Overview & Phase 1 (`.docs/TEAM_277_INSTRUCTIONS_PART_1.md`)
- **Part 2:** Phase 1 & 2 Details (`.docs/TEAM_277_INSTRUCTIONS_PART_2.md`)
- **Part 3:** Phase 3 Details (`.docs/TEAM_277_INSTRUCTIONS_PART_3.md`)
- **Part 4:** Phase 4, 5, 6 Details (`.docs/TEAM_277_INSTRUCTIONS_PART_4.md`)

### 3. Reference Documents (As Needed)
- Design documents (why we're doing this)
- Architecture documents (how rbee works)
- Implementation guides (how to add code)

---

## 🎯 Mission Overview

### Current State (Imperative)
```bash
# User runs commands manually
rbee install-hive --alias gpu-1
rbee install-worker --hive gpu-1 --type vllm
# Sequential, slow, error-prone
```

### Target State (Declarative)
```bash
# User declares config
cat > ~/.config/rbee/hives.conf << 'EOF'
[[hive]]
alias = "gpu-1"
workers = [{ type = "vllm" }]
EOF

# Single command installs everything concurrently
rbee sync
# ✅ Installed 3 hives, 6 workers (30s) - 3x faster!
```

### Key Benefit
- **3-10x faster** (concurrent installation)
- **Desired state management** (config is source of truth)
- **Simpler architecture** (queen manages everything)

---

## 📚 Document Index

### Instructions (Read First)
| Document | Purpose | Read Order |
|----------|---------|------------|
| `TEAM_277_INSTRUCTIONS_PART_1.md` | Overview & Phase 1 | 1st |
| `TEAM_277_INSTRUCTIONS_PART_2.md` | Phase 1 & 2 Details | 2nd |
| `TEAM_277_INSTRUCTIONS_PART_3.md` | Phase 3 Details | 3rd |
| `TEAM_277_INSTRUCTIONS_PART_4.md` | Phase 4, 5, 6 Details | 4th |

### Design Documents (Understand Why)
| Document | Purpose |
|----------|---------|
| `DECLARATIVE_CONFIG_ANALYSIS.md` | Pros/cons of declarative |
| `DECLARATIVE_MIGRATION_PLAN.md` | Full migration strategy (YOUR ROADMAP) |
| `PACKAGE_MANAGER_OPERATIONS.md` | New operations design |

### Architecture Documents (Understand System)
| Document | Purpose |
|----------|---------|
| `.arch/00_OVERVIEW_PART_1.md` | Overall architecture |
| `.arch/01_COMPONENTS_PART_2.md` | Component responsibilities |
| `.arch/03_DATA_FLOW_PART_4.md` | How operations flow |

### Implementation Guides (Understand How)
| Document | Purpose |
|----------|---------|
| `bin/ADDING_NEW_OPERATIONS.md` | 3-file pattern for operations |
| `bin/.plan/TEAM_258_CONSOLIDATION_SUMMARY.md` | Operation routing |
| `bin/.plan/TEAM_259_JOB_CLIENT_CONSOLIDATION.md` | JobClient usage |

### Code to Explore
| Directory | Purpose |
|-----------|---------|
| `bin/05_rbee_keeper_crates/queen-lifecycle/` | Queen lifecycle |
| `bin/15_queen_rbee_crates/hive-lifecycle/` | Hive lifecycle |
| `bin/25_rbee_hive_crates/worker-lifecycle/` | Worker lifecycle |
| `bin/99_shared_crates/daemon-lifecycle/` | Shared utilities |

---

## 🗺️ The 6 Phases

| Phase | Duration | Goal | Status |
|-------|----------|------|--------|
| **Phase 1** | 8-12h | Add config support | ⏳ TODO |
| **Phase 2** | 12-16h | Add package operations | ⏳ TODO |
| **Phase 3** | 24-32h | Implement package manager | ⏳ TODO |
| **Phase 4** | 8-12h | Simplify hive | ⏳ TODO |
| **Phase 5** | 8-12h | Update CLI | ⏳ TODO |
| **Phase 6** | 4-6h | Remove old operations | ⏳ TODO |
| **Total** | **64-90h** | **Declarative lifecycle** | ⏳ TODO |

---

## 🔑 Key Architectural Decision

**Queen manages BOTH hive AND worker installation remotely via SSH**

### Why This Matters

**Old Architecture:**
```
Queen → Install hive (SSH)
Hive → Install workers (local)
```

**New Architecture:**
```
Queen → Install hive (SSH)
Queen → Install workers (SSH)  ← NEW!
Hive → Only manage processes
```

**Benefits:**
- ✅ Simpler hive (no installation logic)
- ✅ Concurrent (queen installs all workers in parallel)
- ✅ Declarative (config declares everything)

---

## 📋 Quick Reference

### New Operations to Add
- `PackageSync` - Sync all to config
- `PackageStatus` - Check drift
- `PackageInstall` - Install all
- `PackageUninstall` - Uninstall all
- `PackageValidate` - Validate config
- `PackageMigrate` - Generate config

### Operations to Remove
- `HiveInstall` → `PackageSync`
- `HiveUninstall` → `PackageUninstall`
- `WorkerDownload` → `PackageSync`
- `WorkerBuild` → `PackageSync`
- `WorkerBinaryList` → `PackageStatus`
- `WorkerBinaryGet` → `PackageStatus`
- `WorkerBinaryDelete` → `PackageSync`

### Operations to Keep (Runtime Management)
- `HiveStart/Stop/List/Get/Status` - Daemon management
- `WorkerSpawn/ProcessList/ProcessGet/ProcessDelete` - Process management
- `ModelDownload/List/Get/Delete` - Model management
- `Infer` - Inference

---

## 🚀 Getting Started

### Step 1: Read Instructions
Start with `TEAM_277_INSTRUCTIONS_PART_1.md`

### Step 2: Read Design Docs
- `DECLARATIVE_CONFIG_ANALYSIS.md`
- `DECLARATIVE_MIGRATION_PLAN.md`
- `PACKAGE_MANAGER_OPERATIONS.md`

### Step 3: Explore Code
- `bin/05_rbee_keeper_crates/queen-lifecycle/`
- `bin/15_queen_rbee_crates/hive-lifecycle/`
- `bin/25_rbee_hive_crates/worker-lifecycle/`

### Step 4: Start Phase 1
Follow `TEAM_277_INSTRUCTIONS_PART_2.md` for detailed steps.

---

## ✅ Success Criteria

**You're done when:**
- ✅ All 6 phases complete
- ✅ `rbee sync` works
- ✅ Concurrent installation works (3-10x faster)
- ✅ Config file is source of truth
- ✅ Old operations removed
- ✅ All tests pass

**Final test:**
```bash
# Create config
cat > ~/.config/rbee/hives.conf << 'EOF'
[[hive]]
alias = "test-hive"
hostname = "localhost"
ssh_user = "vince"
workers = [{ type = "vllm" }]
EOF

# Sync
rbee sync
# ✅ Should install hive + workers concurrently

# Check status
rbee status
# ✅ Should show all components installed
```

---

## 📝 Handoff

When complete, write handoff document:
- **File:** `bin/.plan/TEAM_277_HANDOFF.md`
- **Include:** What you built, what works, what's next
- **Format:** See `bin/.plan/TEAM_212_HANDOFF.md` for example

---

## 🆘 Need Help?

### If Stuck on Phase 1 (Config)
- Read `bin/99_shared_crates/rbee-config/src/lib.rs` for existing config patterns
- Look at TOML examples in `.llorch.toml.example`

### If Stuck on Phase 2 (Operations)
- Read `bin/ADDING_NEW_OPERATIONS.md` for 3-file pattern
- Look at existing operations in `rbee-operations/src/lib.rs`

### If Stuck on Phase 3 (Package Manager)
- Read `bin/15_queen_rbee_crates/hive-lifecycle/src/install.rs` for SSH patterns
- Read `bin/10_queen_rbee/src/hive_forwarder.rs` for concurrent patterns
- Use `tokio::spawn` for concurrency

### If Stuck on Phase 4 (Simplify Hive)
- Just delete files and remove match arms
- Make sure hive only manages processes, not installation

### If Stuck on Phase 5 (CLI)
- Read existing commands in `bin/00_rbee_keeper/src/commands/`
- Follow same pattern

### If Stuck on Phase 6 (Cleanup)
- Just delete old operations
- Update tests

---

## 🎯 Remember

**This is a major architectural improvement:**
- 3-10x faster installation
- Desired state management
- Simpler architecture
- Better for scheduler

**Take your time, test thoroughly, and document everything!**

**Good luck, TEAM-277! 🚀**
