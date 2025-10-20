# bin/ Directory Structure

**Date:** 2025-10-19  
**Purpose:** Document the new numbered directory structure

---

## Overview

The `bin/` directory has been reorganized with numbered prefixes to enforce a clear hierarchy and make the structure self-documenting.

---

## Directory Structure

```
bin/
├── bak.rbee-keeper/          # OLD: CLI tool (to be removed)
├── bak.queen-rbee/           # OLD: Orchestrator daemon (to be removed)
├── bak.rbee-hive/            # OLD: Pool manager daemon (to be removed)
├── bak.llm-worker-rbee/      # OLD: Worker daemon (to be removed)
│
├── 05_rbee_keeper_crates/    # Keeper-specific crates
│   ├── config/
│   ├── cli/
│   ├── commands/
│   └── queen-lifecycle/
│
├── 15_queen_rbee_crates/     # Queen-specific crates
│   ├── ssh-client/
│   ├── hive-registry/
│   ├── worker-registry/
│   ├── hive-lifecycle/
│   ├── http-server/
│   └── preflight/
│
├── 25_rbee_hive_crates/      # Hive-specific crates
│   ├── worker-lifecycle/
│   ├── worker-registry/
│   ├── model-catalog/
│   ├── model-provisioner/
│   ├── monitor/
│   ├── http-server/
│   ├── download-tracker/
│   └── device-detection/
│
├── 39_worker_rbee_crates/    # Worker-specific crates
│   └── http-server/
│
└── 99_shared_crates/         # Shared across all binaries
    ├── daemon-lifecycle/
    ├── rbee-http-client/
    ├── rbee-types/
    ├── heartbeat/
    ├── audit-logging/
    ├── auth-min/
    ├── deadline-propagation/
    ├── input-validation/
    ├── jwt-guardian/
    ├── narration-core/
    ├── narration-macros/
    ├── secrets-management/
    └── model-catalog/
```

---

## Numbering Scheme

### Purpose

The numbering scheme enforces a **dependency hierarchy** and makes the architecture **self-documenting**:

- **Lower numbers** = Higher in the stack (closer to user)
- **Higher numbers** = Lower in the stack (infrastructure)

### Number Ranges

| Range | Purpose | Examples |
|-------|---------|----------|
| `00-09` | **Top-level binaries** | CLI tools, user-facing apps |
| `05-09` | **Keeper crates** | CLI-specific functionality |
| `10-19` | **Queen binaries & crates** | Orchestrator daemon |
| `15-19` | **Queen crates** | Orchestrator-specific functionality |
| `20-29` | **Hive binaries & crates** | Pool manager daemon |
| `25-29` | **Hive crates** | Pool manager-specific functionality |
| `30-39` | **Worker binaries & crates** | Worker daemon |
| `39` | **Worker crates** | Worker-specific functionality |
| `99` | **Shared crates** | Cross-service utilities |

---

## Dependency Rules

### Allowed Dependencies

**Top-down only:**
- ✅ `05_rbee_keeper_crates` → `99_shared_crates`
- ✅ `15_queen_rbee_crates` → `99_shared_crates`
- ✅ `25_rbee_hive_crates` → `99_shared_crates`
- ✅ `39_worker_rbee_crates` → `99_shared_crates`

**Within same level:**
- ✅ `05_rbee_keeper_crates/cli` → `05_rbee_keeper_crates/config`

### Forbidden Dependencies

**Bottom-up (NEVER):**
- ❌ `99_shared_crates` → `05_rbee_keeper_crates`
- ❌ `99_shared_crates` → `15_queen_rbee_crates`
- ❌ `99_shared_crates` → `25_rbee_hive_crates`

**Cross-service (NEVER):**
- ❌ `05_rbee_keeper_crates` → `15_queen_rbee_crates`
- ❌ `15_queen_rbee_crates` → `25_rbee_hive_crates`
- ❌ `25_rbee_hive_crates` → `39_worker_rbee_crates`

---

## Benefits

### 1. Self-Documenting

**Before:**
```
bin/rbee-keeper-crates/
bin/queen-rbee-crates/
bin/rbee-hive-crates/
bin/shared-crates/
```
❌ No clear hierarchy

**After:**
```
bin/05_rbee_keeper_crates/
bin/15_queen_rbee_crates/
bin/25_rbee_hive_crates/
bin/99_shared_crates/
```
✅ Clear hierarchy (05 → 15 → 25 → 99)

### 2. Enforces Architecture

The numbering makes it **visually obvious** when dependencies are wrong:

```rust
// WRONG: Shared crate depending on keeper crate
// bin/99_shared_crates/daemon-lifecycle/Cargo.toml
[dependencies]
rbee-keeper-config = { path = "../../05_rbee_keeper_crates/config" }
```
❌ **99 → 05 is backwards!** (higher number → lower number)

```rust
// CORRECT: Keeper crate depending on shared crate
// bin/05_rbee_keeper_crates/cli/Cargo.toml
[dependencies]
daemon-lifecycle = { path = "../../99_shared_crates/daemon-lifecycle" }
```
✅ **05 → 99 is correct!** (lower number → higher number)

### 3. Alphabetical Sorting

Directories now sort in **dependency order** automatically:

```
bin/
├── 05_rbee_keeper_crates/    # First (top of stack)
├── 15_queen_rbee_crates/     # Second
├── 25_rbee_hive_crates/      # Third
├── 39_worker_rbee_crates/    # Fourth
└── 99_shared_crates/         # Last (bottom of stack)
```

### 4. Room for Growth

**Gaps between numbers allow for future additions:**

- `06-09`: Reserved for future keeper-related crates
- `16-19`: Reserved for future queen-related crates
- `26-29`: Reserved for future hive-related crates
- `40-98`: Reserved for future services

---

## Migration Status

### Completed

- ✅ Renamed all directories
- ✅ Updated `Cargo.toml` workspace members
- ✅ Verified compilation

### TODO

- [ ] Update all path dependencies in crate `Cargo.toml` files
- [ ] Update documentation references
- [ ] Update CI/CD scripts
- [ ] Delete old `bak.*` directories

---

## Related Documentation

- **Architecture:** `bin/.plan/TEAM_130G_FINAL_ARCHITECTURE.md`
- **Consolidation:** `bin/.plan/TEAM_130E_CONSOLIDATION_SUMMARY.md`
- **Deprecations:** `bin/.plan/DEPRECATIONS.md`

---

**Last Updated:** 2025-10-19
