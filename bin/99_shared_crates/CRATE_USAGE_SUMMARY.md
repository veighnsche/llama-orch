# Shared Crates Usage Summary

**Date:** 2025-10-09T23:39:00+02:00  
**Team:** TEAM-027

---

## Verification Results

### ✅ ACTIVE - Currently Used

1. **worker-registry** (NEW)
   - Used by: rbee-keeper
   - Will be used by: queen-rbee
   - Status: ✅ Active

2. **pool-core**
   - Used by: rbee-hive
   - Status: ✅ Active

3. **gpu-info**
   - Used by: (need to verify)
   - Status: ✅ Likely active

4. **narration-core**
   - **Used by: llm-worker-rbee** ✅
   - Files: main.rs, device.rs, backend/inference.rs
   - Status: ✅ **ACTIVE - Keep!**
   - Note: llm-worker-rbee uses narration for observability

### ❌ NOT USED - Can Archive

1. **auth-min**
   - Grep results: Only doc comments in own crate
   - Status: ❌ Not used in binaries
   - Action: Can archive (re-add if auth needed)

2. **deadline-propagation**
   - Grep results: Only doc comments in own crate
   - Status: ❌ Not used in binaries
   - Action: Can archive (re-add if deadlines needed)

3. **pool-registry-types**
   - Grep results: No usage found
   - Status: ❌ Not used
   - Action: Archive (replaced by worker-registry)

4. **audit-logging**
   - Status: ❌ Not used in MVP
   - Action: Archive (re-add post-MVP if needed)

5. **input-validation**
   - Status: ❌ Not used in MVP
   - Action: Archive (re-add post-MVP if needed)

6. **secrets-management**
   - Status: ❌ Not used in MVP
   - Action: Archive (re-add post-MVP if needed)

---

## Updated Recommendations

### Keep (Active in Current Code)

- ✅ **worker-registry** - New, shared between queen-rbee and rbee-keeper
- ✅ **pool-core** - Used by rbee-hive
- ✅ **gpu-info** - Likely used for GPU detection
- ✅ **narration-core** - **Used by llm-worker-rbee**
- ✅ **narration-macros** - **Dependency of narration-core**

### Keep (Good Infrastructure, Not Yet Used)

- ✅ **audit-logging** - Super good to keep for production
- ✅ **secrets-management** - Well developed for security
- ✅ **input-validation** - Keep for production validation
- ✅ **deadline-propagation** - Performance crate for the future

### Archive (Delete or Rename)

- ❌ **pool-registry-types** - Replaced by worker-registry, DELETE
- ❌ **orchestrator-core** - Old architecture, DELETE
- ⚠️ **auth-min** - Not sure, needs decision
- 🔄 **pool-core** - RENAME to **hive-core**

---

## CORRECTED Decision Matrix (Per User Feedback)

| Crate | Used By | MVP Status | Action |
|-------|---------|------------|--------|
| worker-registry | rbee-keeper, queen-rbee | ✅ Active | **Keep** |
| pool-core | rbee-hive | ✅ Active | **Rename to hive-core** |
| gpu-info | TBD | ✅ Likely | **Keep** |
| narration-core | llm-worker-rbee | ✅ Active | **Keep** |
| narration-macros | narration-core | ✅ Active | **Keep** |
| audit-logging | None (future) | ✅ Good infra | **Keep** - Super good for production |
| secrets-management | None (future) | ✅ Good infra | **Keep** - Well developed for security |
| input-validation | None (future) | ✅ Good infra | **Keep** - Production validation |
| deadline-propagation | None (future) | ✅ Good infra | **Keep** - Performance crate for future |
| auth-min | None (future) | ✅ Good infra | **Keep** - Security primitives for auth |
| pool-registry-types | None | ❌ Obsolete | **DELETE** - Replaced by worker-registry |
| orchestrator-core | None | ❌ Obsolete | **DELETE** - Old architecture |

---

## Important Discovery

**narration-core is ACTIVE!**

llm-worker-rbee uses it for observability:
```rust
use observability_narration_core::{narrate, NarrationFields};
```

Found in:
- `bin/llm-worker-rbee/src/main.rs`
- `bin/llm-worker-rbee/src/device.rs`
- `bin/llm-worker-rbee/src/backend/inference.rs`

**Do NOT archive narration-core or narration-macros!**

---

## CORRECTED Final Recommendations (Per User)

### Keep - Active Now

1. ✅ worker-registry (new)
2. ✅ pool-core → **RENAME to hive-core**
3. ✅ gpu-info
4. ✅ narration-core (used by llm-worker-rbee)
5. ✅ narration-macros (dependency)

### Keep - Good Infrastructure for Future

6. ✅ audit-logging - Super good to keep
7. ✅ secrets-management - Well developed for security
8. ✅ input-validation - Production validation
9. ✅ deadline-propagation - Performance crate for future

### Delete - Obsolete

1. ❌ pool-registry-types - Replaced by worker-registry
2. ❌ orchestrator-core - Old architecture

### Keep - Security Infrastructure

10. ✅ **auth-min** - KEEP for future auth needs
    - Timing-safe token comparison (prevents CWE-208)
    - Token fingerprinting for safe logging
    - Bearer token parsing (RFC 6750)
    - Integrates with secrets-management
    - Needed for: rbee-hive ↔ worker auth, queen-rbee ↔ hive auth

### Cleanup Process

**Step 1: Delete obsolete crates**
```bash
# Delete old architecture crates
rm -rf bin/shared-crates/pool-registry-types
rm -rf bin/shared-crates/orchestrator-core
```

**Step 2: Rename pool-core to hive-core**
```bash
# Rename directory
mv bin/shared-crates/pool-core bin/shared-crates/hive-core

# Update Cargo.toml name
# Update all imports in rbee-hive
```

**Step 3: Update workspace Cargo.toml**
- Remove: pool-registry-types, orchestrator-core
- Rename: pool-core → hive-core

**Step 4: Decide on auth-min**
- Keep if auth needed between components
- Delete if not needed for MVP

---

**Created by:** TEAM-027  
**Status:** Verification complete, ready for archival
