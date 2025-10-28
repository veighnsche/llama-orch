# TEAM-329: Contract Entropy - daemon-contract

**Date:** 2025-10-27  
**Rule:** RULE ZERO - Don't create "reusable" abstractions with 1 consumer

## The Problem

**daemon-contract exists as a separate crate but has ONLY ONE consumer: daemon-lifecycle**

### Evidence

```bash
# Find all Cargo.toml files that depend on daemon-contract
$ rg "daemon-contract" --type toml -g "Cargo.toml"

# Result: ONLY daemon-lifecycle uses it
bin/99_shared_crates/daemon-lifecycle/Cargo.toml:daemon-contract = { path = "../../97_contracts/daemon-contract" }
```

### What's in daemon-contract?

```
daemon-contract/
├── handle.rs         - DaemonHandle (generic handle)
├── install.rs        - InstallConfig, InstallResult, UninstallConfig
├── lifecycle.rs      - HttpDaemonConfig (168 LOC!)
├── shutdown.rs       - ShutdownConfig
└── status.rs         - StatusRequest, StatusResponse
```

**Total:** ~400 LOC of "contracts" used by exactly 1 crate.

## The Entropy

### Classic Over-Engineering Pattern

1. **Created "contract" crate** for "reusability"
2. **Only 1 consumer** ever used it (daemon-lifecycle)
3. **No other consumers** in 2+ years
4. **Extra indirection** for zero benefit

### Why It's Entropy

**Contracts are for MULTIPLE consumers:**
- API contracts (HTTP/gRPC) → multiple clients
- Database schemas → multiple services
- Message formats → multiple producers/consumers

**daemon-contract has 1 consumer:**
- daemon-lifecycle is the ONLY user
- No other crate imports it
- No external consumers
- No API boundary

**Result:** Unnecessary layer of indirection.

## The Fix

**INLINE daemon-contract into daemon-lifecycle**

### Step 1: Move Files

```bash
# Move all contract files into daemon-lifecycle
mv bin/97_contracts/daemon-contract/src/*.rs \
   bin/99_shared_crates/daemon-lifecycle/src/types/

# Update module structure
daemon-lifecycle/
├── types/
│   ├── mod.rs           (NEW)
│   ├── handle.rs        (from daemon-contract)
│   ├── install.rs       (from daemon-contract)
│   ├── lifecycle.rs     (from daemon-contract)
│   ├── shutdown.rs      (from daemon-contract)
│   └── status.rs        (from daemon-contract)
```

### Step 2: Update Imports

```rust
// OLD (entropy)
use daemon_contract::HttpDaemonConfig;
use daemon_contract::{InstallConfig, InstallResult};

// NEW (clean)
use crate::types::{HttpDaemonConfig, InstallConfig, InstallResult};
```

### Step 3: Delete daemon-contract Crate

```bash
rm -rf bin/97_contracts/daemon-contract
```

### Step 4: Update Cargo.toml

```toml
# Remove dependency
# daemon-contract = { path = "../../97_contracts/daemon-contract" }
```

## Why This Matters

### Before (Entropy)
```
daemon-contract (separate crate, 400 LOC)
    ↓ (only consumer)
daemon-lifecycle (uses contracts)
```

**Cost:**
- Extra crate to maintain
- Extra Cargo.toml
- Extra compilation unit
- Extra indirection
- Misleading name (implies multiple consumers)

### After (Clean)
```
daemon-lifecycle
    └── types/ (inline, 400 LOC)
```

**Benefits:**
- ✅ Single crate
- ✅ No indirection
- ✅ Honest about scope (internal types, not contracts)
- ✅ Easier to refactor (no cross-crate changes)

## Decision Matrix

| Scenario | Create Contract Crate? |
|----------|----------------------|
| 1 consumer | ❌ NO - inline it |
| 2 consumers | ⚠️ MAYBE - wait and see |
| 3+ consumers | ✅ YES - extract contract |
| External API | ✅ YES - stable contract |

**daemon-contract:** 1 consumer → ❌ INLINE IT

## Historical Context

**Why was it created?**

TEAM-315/316 comment: "Generic daemon lifecycle contracts... for consistent lifecycle management across the rbee ecosystem"

**Intended consumers:**
- queen-rbee (manages hives)
- rbee-hive (manages workers)
- rbee-keeper (manages queen)

**Reality:**
- Only daemon-lifecycle uses it
- queen-rbee, rbee-hive, rbee-keeper don't import it
- They use daemon-lifecycle functions, not contracts directly

**Lesson:** Don't create "reusable" abstractions before you have 2+ consumers.

## Comparison: health.rs

**health.rs does it RIGHT:**

```rust
// health.rs - inline types
pub struct HealthPollConfig { ... }

// Re-export contract types only when needed
pub use daemon_contract::{StatusRequest, StatusResponse};
```

**Why StatusRequest/StatusResponse are OK:**
- They're HTTP API types (actual contract boundary)
- Multiple services might implement the status endpoint
- Serialization format matters (wire protocol)

**Why InstallConfig/HttpDaemonConfig are NOT:**
- Internal configuration types
- Only used within daemon-lifecycle
- No wire protocol
- No external consumers

## Next Steps

1. ✅ **Document the problem** (this file)
2. ⚠️ **Plan the migration** (inline daemon-contract)
3. ⚠️ **Execute migration** (move files, update imports)
4. ⚠️ **Delete daemon-contract** (remove crate)
5. ⚠️ **Verify compilation** (all tests pass)

---

**Key Insight:** Contracts are for BOUNDARIES. If there's no boundary (only 1 consumer), it's not a contract - it's just types that should be inline.

**RULE ZERO:** Don't create "reusable" abstractions until you have 2+ consumers. One consumer = inline it.
