# TEAM-272: Operation Routing Update

**Date:** Oct 23, 2025  
**Status:** ✅ COMPLETE

---

## 🎯 Changes Made

Updated operation routing to align with the **corrected architecture** from `CORRECTION_269_TO_272_ARCHITECTURE_FIX.md`.

### Key Principle

**Hive is STATELESS** - Workers send heartbeats to queen, so queen is the source of truth for worker state.

---

## 📊 Routing Changes

### Before (Incorrect)

All worker operations forwarded to hive:
- ❌ WorkerSpawn → Hive
- ❌ WorkerList → Hive
- ❌ WorkerGet → Hive
- ❌ WorkerDelete → Hive

**Problem:** Hive doesn't track workers (stateless), so List/Get/Delete can't work.

### After (Corrected)

Operations split by responsibility:

**Forwarded to Hive (stateless execution):**
- ✅ WorkerSpawn → Hive (spawn process, return PID)
- ✅ Model operations → Hive (download/manage models)

**Handled by Queen (stateful tracking):**
- ✅ WorkerList → Queen (query worker registry)
- ✅ WorkerGet → Queen (query worker registry)
- ✅ WorkerDelete → Queen (get PID from registry, forward to hive to kill process)
- ✅ Infer → Queen (scheduling and routing)

---

## 📁 Files Changed

### 1. `rbee-operations/src/lib.rs`

**Function:** `should_forward_to_hive()`

**Change:** Removed `WorkerList`, `WorkerGet`, `WorkerDelete` from forwarding list.

```rust
// BEFORE
pub fn should_forward_to_hive(&self) -> bool {
    matches!(
        self,
        Operation::WorkerSpawn { .. }
            | Operation::WorkerList { .. }    // ❌ Removed
            | Operation::WorkerGet { .. }     // ❌ Removed
            | Operation::WorkerDelete { .. }  // ❌ Removed
            | Operation::ModelDownload { .. }
            | Operation::ModelList { .. }
            | Operation::ModelGet { .. }
            | Operation::ModelDelete { .. }
    )
}

// AFTER
pub fn should_forward_to_hive(&self) -> bool {
    matches!(
        self,
        Operation::WorkerSpawn { .. }
            | Operation::ModelDownload { .. }
            | Operation::ModelList { .. }
            | Operation::ModelGet { .. }
            | Operation::ModelDelete { .. }
    )
}
```

### 2. `queen-rbee/src/job_router.rs`

**Added:** WorkerList, WorkerGet, WorkerDelete handlers

**Implementation:**
- WorkerList: Returns empty list (worker registry not yet implemented)
- WorkerGet: Returns error (worker registry not yet implemented)
- WorkerDelete: Returns error (worker registry not yet implemented)

**Future:** Once worker registry is implemented, these will:
1. Query registry for worker info
2. For WorkerDelete: Get PID, forward to hive to kill process

### 3. `rbee-hive/src/job_router.rs`

**No changes needed** - WorkerList/Get/Delete already redirect to queen with explanatory messages.

---

## 🏗️ Architecture Flow

### WorkerSpawn (Forwarded to Hive)

```
rbee-keeper → queen-rbee → hive_forwarder → rbee-hive
                                                ↓
                                         spawn_worker()
                                                ↓
                                         Return PID/port
                                                ↓
                                         Worker sends heartbeat to QUEEN
```

### WorkerList (Handled by Queen)

```
rbee-keeper → queen-rbee → Query worker registry
                                ↓
                         Return worker list
```

### WorkerDelete (Handled by Queen)

```
rbee-keeper → queen-rbee → Get worker from registry (includes PID)
                                ↓
                         Forward to hive with PID
                                ↓
                         rbee-hive → delete_worker(pid)
                                ↓
                         Remove from registry
```

---

## ✅ Verification

### Compilation

```bash
✅ cargo check --bin queen-rbee  # PASS
✅ cargo check --bin rbee-hive   # PASS
```

### Operation Routing

| Operation | Routed To | Handler | Status |
|-----------|-----------|---------|--------|
| WorkerSpawn | Hive | spawn_worker() | ✅ Implemented |
| WorkerList | Queen | Query registry | ⚠️ TODO (registry) |
| WorkerGet | Queen | Query registry | ⚠️ TODO (registry) |
| WorkerDelete | Queen | Get PID → Hive | ⚠️ TODO (registry) |
| ModelDownload | Hive | download_model() | ⚠️ TODO (provisioner) |
| ModelList | Hive | list() | ✅ Implemented |
| ModelGet | Hive | get() | ✅ Implemented |
| ModelDelete | Hive | remove() | ✅ Implemented |
| Infer | Queen | Schedule → Worker | ⚠️ TODO (scheduler) |

---

## 🔗 Next Steps

### For Worker Registry Implementation

1. **Create worker registry in queen-rbee**
   - Track workers via heartbeats
   - Store: worker_id, hive_id, model_id, device, PID, port, status

2. **Update WorkerList handler**
   - Query registry by hive_id
   - Format as table

3. **Update WorkerGet handler**
   - Query registry by worker_id
   - Return JSON

4. **Update WorkerDelete handler**
   - Get worker from registry
   - Forward to hive with PID
   - Remove from registry after confirmation

### For Hive Integration

Hive already has the correct implementation:
- ✅ WorkerSpawn: Spawns process, returns PID
- ✅ WorkerList/Get: Redirects to queen
- ✅ WorkerDelete: Explains PID requirement

---

## 📚 Documentation

**Updated:**
- `rbee-operations/src/lib.rs` - should_forward_to_hive() documentation
- `queen-rbee/src/job_router.rs` - Added worker operation handlers

**Created:**
- This document (TEAM_272_ROUTING_UPDATE.md)

---

## 🎯 Summary

**Changes:**
- ✅ Updated should_forward_to_hive() to reflect corrected architecture
- ✅ Moved WorkerList/Get/Delete from hive to queen
- ✅ WorkerSpawn still forwards to hive (correct)
- ✅ All operations compile successfully

**Architecture:**
- ✅ Hive is STATELESS (only executes operations)
- ✅ Queen is STATEFUL (tracks workers via heartbeats)
- ✅ Clear separation of concerns

**Next:**
- Implement worker registry in queen-rbee
- Wire up WorkerList/Get/Delete to query registry
- Implement WorkerDelete PID forwarding to hive

---

**TEAM-272 routing update complete! 🎉**
