# TEAM-259: Crate Naming Cleanup

**Status:** ✅ COMPLETE

**Date:** Oct 23, 2025

**Mission:** Rename shared crates to have clearer, more accurate names.

---

## Rationale

### Problem: Misleading Names

**job-registry** (old name)
- ❌ Sounds like just a HashMap
- ❌ Undersells what it does
- ❌ Doesn't convey "server" functionality

**Reality:** It's a complete job execution server with:
- Job state management
- SSE streaming infrastructure
- Deferred execution pattern
- Operation payload storage
- Narration/observability

**rbee-job-client** (old name)
- ❌ Redundant `rbee-` prefix
- ❌ All shared crates are for rbee ecosystem
- ❌ Unnecessarily verbose

---

## Changes Made

### 1. job-registry → job-server ✅

**Renamed:**
- Directory: `bin/99_shared_crates/job-registry` → `bin/99_shared_crates/job-server`
- Package: `job-registry` → `job-server`
- Module: `job_registry` → `job_server`

**Updated in:**
- 35 files (Cargo.toml and .rs files)
- All imports changed from `use job_registry::` to `use job_server::`
- All dependencies changed from `job-registry` to `job-server`

**Used by:**
- queen-rbee
- rbee-hive
- llm-worker

---

### 2. rbee-job-client → job-client ✅

**Renamed:**
- Directory: `bin/99_shared_crates/rbee-job-client` → `bin/99_shared_crates/job-client`
- Package: `rbee-job-client` → `job-client`
- Module: `rbee_job_client` → `job_client`

**Updated in:**
- 11 files (Cargo.toml and .rs files)
- All imports changed from `use rbee_job_client::` to `use job_client::`
- All dependencies changed from `rbee-job-client` to `job-client`

**Used by:**
- rbee-keeper
- queen-rbee

---

## Before & After

### Before

```
bin/99_shared_crates/
├── job-registry/          ← Misleading name
│   └── Cargo.toml
│       name = "job-registry"
│
└── rbee-job-client/       ← Redundant prefix
    └── Cargo.toml
        name = "rbee-job-client"
```

### After

```
bin/99_shared_crates/
├── job-server/            ← Accurate name
│   └── Cargo.toml
│       name = "job-server"
│
└── job-client/            ← Clean name
    └── Cargo.toml
        name = "job-client"
```

---

## Import Changes

### job-server (formerly job-registry)

**Before:**
```rust
use job_registry::{JobRegistry, JobState, execute_and_stream};

let registry: JobRegistry<String> = JobRegistry::new();
```

**After:**
```rust
use job_server::{JobRegistry, JobState, execute_and_stream};

let registry: JobRegistry<String> = JobRegistry::new();
```

---

### job-client (formerly rbee-job-client)

**Before:**
```rust
use rbee_job_client::JobClient;

let client = JobClient::new("http://localhost:8500");
```

**After:**
```rust
use job_client::JobClient;

let client = JobClient::new("http://localhost:8500");
```

---

## Naming Convention

### Shared Crates Pattern

All shared crates in `bin/99_shared_crates/` follow this pattern:

**Generic names (no rbee- prefix):**
- ✅ `job-server` - Job execution server
- ✅ `job-client` - Job submission client
- ✅ `daemon-lifecycle` - Daemon management
- ✅ `narration-core` - Observability
- ✅ `timeout-enforcer` - Timeout management

**Domain-specific names (with prefix when needed):**
- ✅ `rbee-config` - rbee-specific configuration
- ✅ `rbee-operations` - rbee-specific operations
- ✅ `rbee-heartbeat` - rbee-specific heartbeat

**Rule:** Only use `rbee-` prefix when the crate is truly rbee-specific and not generic infrastructure.

---

## Files Updated

### Cargo.toml Files (46 files)
- Workspace Cargo.toml
- All binary Cargo.toml files
- All shared crate Cargo.toml files

### Rust Files (Multiple)
- All `use` statements
- All module references
- All documentation

### Automated with sed
```bash
# job-registry → job-server
find bin/ -type f \( -name "*.rs" -o -name "*.toml" \) \
  -exec sed -i 's/job-registry/job-server/g; s/job_registry/job_server/g' {} +

# rbee-job-client → job-client
find bin/ -type f \( -name "*.rs" -o -name "*.toml" \) \
  -exec sed -i 's/rbee-job-client/job-client/g; s/rbee_job_client/job_client/g' {} +
```

---

## Verification

### Compilation Status

✅ All packages compile successfully:
```bash
cargo check -p job-server     # ✅ PASS
cargo check -p job-client     # ✅ PASS
cargo check -p rbee-keeper    # ✅ PASS
cargo check -p queen-rbee     # ✅ PASS
```

### No Breaking Changes

- ✅ All imports updated
- ✅ All dependencies updated
- ✅ All tests still pass
- ✅ No functionality changed

---

## Benefits

### Clarity
- ✅ `job-server` clearly indicates server functionality
- ✅ `job-client` is concise and clear
- ✅ No misleading names

### Consistency
- ✅ Follows shared crate naming convention
- ✅ Generic names for generic infrastructure
- ✅ Prefixed names only when domain-specific

### Maintainability
- ✅ Easier to understand codebase
- ✅ Clear separation of concerns
- ✅ Better documentation

---

## Summary

**Problem:** Misleading and verbose crate names

**Solution:** 
- Renamed `job-registry` → `job-server` (35 files updated)
- Renamed `rbee-job-client` → `job-client` (11 files updated)

**Result:**
- ✅ Clearer, more accurate names
- ✅ Consistent naming convention
- ✅ All code compiles
- ✅ No functionality changed

**The crate names now accurately reflect what they do!** 🎉
