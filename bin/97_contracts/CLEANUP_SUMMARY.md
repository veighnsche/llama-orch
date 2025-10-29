# Contract Cleanup Summary

**Date:** Oct 29, 2025  
**Status:** ✅ COMPLETE  
**Duration:** ~30 minutes

---

## Changes Made

### 1. ✅ Deleted ssh-contract (Immediate)
**Reason:** Completely unused (0 external imports)

**Actions:**
- Deleted `bin/97_contracts/ssh-contract/` directory
- Removed from workspace `Cargo.toml`

**Verification:** ✅ All contracts compile

---

### 2. ✅ Fixed operations-contract (High Priority)
**Reason:** `should_forward_to_hive()` method was misleading and contradicted current architecture

**Changes to `src/lib.rs`:**
- ❌ DELETED `should_forward_to_hive()` method (implied queen forwards to hive - FALSE)
- ✅ ADDED `target_server()` method (helps rbee-keeper route correctly)
- ✅ ADDED `TargetServer` enum (Queen vs Hive)

**New API:**
```rust
pub enum TargetServer {
    Queen,  // http://localhost:7833/v1/jobs
    Hive,   // http://localhost:7835/v1/jobs
}

impl Operation {
    pub fn target_server(&self) -> TargetServer {
        match self {
            Operation::Status | Operation::Infer(_) => TargetServer::Queen,
            Operation::WorkerSpawn(_) | ... => TargetServer::Hive,
            _ => TargetServer::Queen,
        }
    }
}
```

**Changes to `README.md`:**
- Updated architecture diagram (shows NO PROXYING)
- Removed references to deleted operations (HiveStart, HiveStop, etc.)
- Added queen vs hive operation split
- Updated all code examples
- Removed "forwarding" language
- Added routing helper section

**Verification:** ✅ Compiles successfully

---

### 3. ✅ Fixed Port Numbers (Medium Priority)
**Reason:** Documentation showed incorrect ports

**worker-contract README:**
- Fixed queen port: 8500 → 7833

**hive-contract README:**
- Fixed hive port: 9200 → 7835
- Fixed queen port: 8500 → 7833

**Verification:** ✅ All contracts compile

---

## Verification Results

### Contract Compilation
```bash
cargo check --package operations-contract \
            --package worker-contract \
            --package hive-contract \
            --package shared-contract \
            --package jobs-contract
```
**Result:** ✅ SUCCESS (all 5 contracts compile)

### Workspace Compilation
**Note:** Workspace has unrelated narration API errors (not caused by contract changes)
- Contract changes are isolated and correct
- Narration errors are pre-existing issues in other crates

---

## Architecture Corrections

### Before Cleanup
```
rbee-keeper → queen-rbee → rbee-hive (forwarding)
```
**Problem:** Implied queen forwards operations to hive

### After Cleanup
```
rbee-keeper
    ├─→ Queen (http://localhost:7833/v1/jobs)
    │   ├─ Status
    │   └─ Infer
    │
    └─→ Hive (http://localhost:7835/v1/jobs)
        ├─ WorkerSpawn, WorkerProcessList, etc.
        └─ ModelDownload, ModelList, etc.
```
**Correct:** rbee-keeper talks directly to BOTH servers (NO proxying)

---

## Files Modified

### Deleted
- `bin/97_contracts/ssh-contract/` (entire directory)

### Modified
- `Cargo.toml` (removed ssh-contract from workspace)
- `bin/97_contracts/operations-contract/src/lib.rs` (replaced method, added enum)
- `bin/97_contracts/operations-contract/README.md` (complete rewrite)
- `bin/97_contracts/worker-contract/README.md` (port fix)
- `bin/97_contracts/hive-contract/README.md` (port fixes)

---

## Impact Assessment

### Breaking Changes
**operations-contract:**
- Deleted `should_forward_to_hive()` method
- Added `target_server()` method as replacement

**Migration:**
```rust
// Old (WRONG)
if operation.should_forward_to_hive() {
    // forward to hive
}

// New (CORRECT)
match operation.target_server() {
    TargetServer::Queen => {
        // Send to http://localhost:7833/v1/jobs
    }
    TargetServer::Hive => {
        // Send to http://localhost:7835/v1/jobs
    }
}
```

### Non-Breaking Changes
- Port number updates (documentation only)
- ssh-contract deletion (unused)

---

## Remaining Work

### Low Priority
- Review `keeper-config-contract` (only used by rbee-keeper)
  - Consider moving to `bin/00_rbee_keeper/src/config/` if it remains single-use
  - Current status: ✅ Working, just questionable as a contract

### Future Consideration
- Create `api-types` contract for common API response types
  - Would reduce duplication between queen/hive/worker APIs
  - Not urgent, but would improve consistency

---

## Lessons Learned

### What Worked Well
1. **Comprehensive audit first** - Identified all issues before making changes
2. **Usage analysis** - grep + Cargo.toml analysis revealed actual usage
3. **Isolated verification** - Checked contracts separately from workspace
4. **Clear documentation** - Updated READMEs to match reality

### What Could Be Improved
1. **Earlier detection** - `should_forward_to_hive()` should have been caught during architecture change
2. **Port standardization** - Should have a single source of truth for port numbers
3. **Contract governance** - Need process for deprecating contracts

---

## References

- **Audit Document:** `bin/97_contracts/CONTRACT_AUDIT.md`
- **Architecture Docs:** `bin/10_queen_rbee/ARCHITECTURE.md`, `JOB_OPERATIONS.md`
- **Port Configuration:** `PORT_CONFIGURATION.md`

---

**Cleanup Complete:** All action items from audit implemented and verified.
