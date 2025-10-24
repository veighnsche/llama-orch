# TEAM-278 DELETION PLAN - Break Everything First

**Date:** Oct 23, 2025  
**Philosophy:** DELETE FIRST, BUILD SECOND

## 🔥 The Problem

Teams keep adding new code without deleting old code = ENTROPY.

## ✅ The Solution

**BREAK THE CODE FIRST.** Delete ALL old operations. Nothing will compile. Nothing will work. THEN rebuild with new operations.

---

## Phase 1: DELETE Old Operations (TEAM-278 NOW)

### Operations to DELETE from `rbee-operations/src/lib.rs`

**DELETE these variants from Operation enum:**

```rust
// ❌ DELETE - Replaced by PackageSync
HiveInstall { alias: String },
HiveUninstall { alias: String },

// ❌ DELETE - Replaced by PackageSync  
WorkerDownload { hive_id: String, worker_type: String },
WorkerBuild { hive_id: String, worker_type: String },
WorkerBinaryList { hive_id: String },
WorkerBinaryGet { hive_id: String, worker_type: String },
WorkerBinaryDelete { hive_id: String, worker_type: String },

// ❌ DELETE - Not needed in declarative arch
SshTest { alias: String },
HiveImportSsh { ssh_config_path: String, default_hive_port: u16 },
```

**Total: 10 operations DELETED**

### What to DELETE from `Operation::name()`

```rust
// DELETE these match arms:
Operation::SshTest { .. } => "ssh_test",
Operation::HiveInstall { .. } => "hive_install",
Operation::HiveUninstall { .. } => "hive_uninstall",
Operation::HiveImportSsh { .. } => "hive_import_ssh",
Operation::WorkerDownload { .. } => "worker_download",
Operation::WorkerBuild { .. } => "worker_build",
Operation::WorkerBinaryList { .. } => "worker_binary_list",
Operation::WorkerBinaryGet { .. } => "worker_binary_get",
Operation::WorkerBinaryDelete { .. } => "worker_binary_delete",
```

### What to DELETE from `Operation::hive_id()`

```rust
// DELETE these match arms:
Operation::HiveInstall { alias } => Some(alias),
Operation::HiveUninstall { alias } => Some(alias),
Operation::WorkerDownload { hive_id, .. } => Some(hive_id),
Operation::WorkerBuild { hive_id, .. } => Some(hive_id),
Operation::WorkerBinaryList { hive_id } => Some(hive_id),
Operation::WorkerBinaryGet { hive_id, .. } => Some(hive_id),
Operation::WorkerBinaryDelete { hive_id, .. } => Some(hive_id),
```

### What to DELETE from `Operation::should_forward_to_hive()`

```rust
// DELETE these from matches!:
Operation::WorkerDownload { .. }
| Operation::WorkerBuild { .. }
| Operation::WorkerBinaryList { .. }
| Operation::WorkerBinaryGet { .. }
| Operation::WorkerBinaryDelete { .. }
```

### What to DELETE from constants

```rust
// DELETE from pub mod constants:
pub const OP_HIVE_INSTALL: &str = "hive_install";
pub const OP_HIVE_UNINSTALL: &str = "hive_uninstall";
```

### What to DELETE from tests

```rust
// DELETE these test functions:
test_serialize_hive_install()
test_serialize_hive_install_remote()
test_serialize_hive_uninstall()
test_operation_name() // Update to remove deleted operations
test_operation_hive_id() // Update to remove deleted operations
```

---

## Expected Result After Phase 1

**EVERYTHING WILL BREAK:**

- ❌ `cargo check -p rbee-operations` will FAIL (tests reference deleted operations)
- ❌ `cargo check -p queen-rbee` will FAIL (job_router.rs has match arms for deleted operations)
- ❌ `cargo check -p rbee-keeper` will FAIL (CLI has commands for deleted operations)
- ❌ `cargo check -p rbee-hive` will FAIL (handles deleted operations)

**THIS IS GOOD. THIS IS WHAT WE WANT.**

---

## Phase 2: Fix Compilation Errors (TEAM-279)

After deletion, TEAM-279 will:

1. **Add 6 new package operations** to replace deleted ones
2. **Remove match arms** from job_router.rs for deleted operations
3. **Remove CLI commands** from rbee-keeper for deleted operations
4. **Remove handlers** from rbee-hive for deleted operations

**Only THEN will code compile again.**

---

## Operations That STAY (Don't Delete)

### Hive Daemon Management (Runtime)
- ✅ `HiveStart` - Start hive daemon
- ✅ `HiveStop` - Stop hive daemon
- ✅ `HiveList` - List configured hives
- ✅ `HiveGet` - Get hive details
- ✅ `HiveStatus` - Check hive health
- ✅ `HiveRefreshCapabilities` - Refresh GPU info

### Worker Process Management (Runtime)
- ✅ `WorkerSpawn` - Spawn worker process
- ✅ `WorkerProcessList` - List running workers
- ✅ `WorkerProcessGet` - Get worker details
- ✅ `WorkerProcessDelete` - Kill worker process

### Active Worker Registry (Runtime)
- ✅ `ActiveWorkerList` - List workers sending heartbeats
- ✅ `ActiveWorkerGet` - Get active worker details
- ✅ `ActiveWorkerRetire` - Retire active worker

### Model Management (On-Demand)
- ✅ `ModelDownload` - Download model
- ✅ `ModelList` - List models
- ✅ `ModelGet` - Get model details
- ✅ `ModelDelete` - Delete model

### Inference
- ✅ `Infer` - Run inference

### System
- ✅ `Status` - System status

**Total: 21 operations KEPT**

---

## New Operations to ADD (Phase 2)

### Package Management (Declarative)
- 🆕 `PackageSync` - Sync all to config
- 🆕 `PackageStatus` - Check drift from config
- 🆕 `PackageInstall` - Install all from config
- 🆕 `PackageUninstall` - Uninstall all
- 🆕 `PackageValidate` - Validate config
- 🆕 `PackageMigrate` - Generate config from current state

**Total: 6 operations ADDED**

---

## Summary

**DELETE:** 10 operations  
**KEEP:** 21 operations  
**ADD:** 6 operations  
**FINAL:** 27 operations total

**Net change:** -4 operations (simpler!)

---

## Execution Order

1. **NOW (TEAM-278):** DELETE all 10 old operations from rbee-operations/src/lib.rs
2. **NEXT (TEAM-279):** ADD 6 new package operations
3. **THEN (TEAM-280):** Implement package manager in queen-rbee
4. **THEN (TEAM-281):** Remove deleted operation handlers from rbee-hive
5. **THEN (TEAM-282):** Remove deleted CLI commands from rbee-keeper
6. **FINALLY (TEAM-283):** Verify everything compiles and works

---

**Ready to DELETE? Let's break everything.**
