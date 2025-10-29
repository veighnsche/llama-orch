# TEAM-272: Complete Operation Architecture

**Date:** Oct 23, 2025  
**Status:** ✅ COMPLETE

---

## 🎯 Operation Routing Architecture

### **HIVE (Stateless Executor)**

Hive manages **local resources** on the hive machine:

#### Worker Binary Operations
- `WorkerDownload` - Download worker binary to hive
- `WorkerBuild` - Build worker binary on hive
- `WorkerBinaryList` - List worker binaries available on hive
- `WorkerBinaryGet` - Get details of a worker binary
- `WorkerBinaryDelete` - Delete worker binary from hive

#### Worker Process Operations (Local)
- `WorkerSpawn` - Spawn a worker process on hive
- `WorkerProcessList` - List worker processes running on hive (local `ps`)
- `WorkerProcessGet` - Get details of a worker process (local `ps`)
- `WorkerProcessDelete` - Kill a worker process on hive

#### Model Operations
- `ModelDownload` - Download model to hive
- `ModelList` - List models on hive
- `ModelGet` - Get model details
- `ModelDelete` - Delete model from hive

**Total Hive Operations:** 13

---

### **QUEEN (Orchestrator)**

Queen manages **distributed state** and orchestration:

#### Hive Operations (Config-based)
- `HiveList` - List hives from `~/.config/rbee/hives.conf`
- `HiveGet` - Get hive details from config
- `HiveInstall` - Install rbee-hive on remote machine
- `HiveUninstall` - Uninstall rbee-hive
- `HiveStart` - Start rbee-hive daemon
- `HiveStop` - Stop rbee-hive daemon
- `HiveStatus` - Check hive health endpoint
- `HiveRefreshCapabilities` - Refresh device capabilities
- `SshTest` - Test SSH connection to hive
- `HiveImportSsh` - Import SSH config into hives.conf

#### Active Worker Operations (Registry-based)
- `ActiveWorkerList` - List active workers (from heartbeat registry)
- `ActiveWorkerGet` - Get details of an active worker
- `ActiveWorkerRetire` - Retire an active worker (stop accepting requests)

#### Inference Operations
- `Infer` - Route inference request to active worker (with scheduling)

#### System Operations
- `Status` - Show live status of all hives and workers

**Total Queen Operations:** 15

---

## 📊 Complete Operation List

### Hive Operations (Forwarded to Hive)

| Operation | Hive ID | Description |
|-----------|---------|-------------|
| **Worker Binaries** |||
| `WorkerDownload` | ✅ | Download worker binary (e.g., cuda-llm-worker) |
| `WorkerBuild` | ✅ | Build worker binary from source |
| `WorkerBinaryList` | ✅ | List available worker binaries |
| `WorkerBinaryGet` | ✅ | Get worker binary details |
| `WorkerBinaryDelete` | ✅ | Delete worker binary |
| **Worker Processes** |||
| `WorkerSpawn` | ✅ | Spawn worker process |
| `WorkerProcessList` | ✅ | List running worker processes (local ps) |
| `WorkerProcessGet` | ✅ | Get worker process details (local ps) |
| `WorkerProcessDelete` | ✅ | Kill worker process |
| **Models** |||
| `ModelDownload` | ✅ | Download model |
| `ModelList` | ✅ | List models |
| `ModelGet` | ✅ | Get model details |
| `ModelDelete` | ✅ | Delete model |

### Queen Operations (Handled by Queen)

| Operation | Hive ID | Description |
|-----------|---------|-------------|
| **System** |||
| `Status` | ❌ | Live status of all hives and workers |
| **Hive Management** |||
| `SshTest` | ✅ | Test SSH connection |
| `HiveInstall` | ✅ | Install rbee-hive |
| `HiveUninstall` | ✅ | Uninstall rbee-hive |
| `HiveStart` | ✅ | Start rbee-hive daemon |
| `HiveStop` | ✅ | Stop rbee-hive daemon |
| `HiveList` | ❌ | List hives from config |
| `HiveGet` | ✅ | Get hive details from config |
| `HiveStatus` | ✅ | Check hive health |
| `HiveRefreshCapabilities` | ✅ | Refresh device capabilities |
| `HiveImportSsh` | ❌ | Import SSH config |
| **Active Workers** |||
| `ActiveWorkerList` | ❌ | List active workers (heartbeat registry) |
| `ActiveWorkerGet` | ❌ | Get active worker details |
| `ActiveWorkerRetire` | ❌ | Retire active worker |
| **Inference** |||
| `Infer` | ✅ | Route inference to active worker |

---

## 🏗️ Architecture Flow

### Worker Binary Management (Hive-Local)

```
rbee-keeper → queen-rbee → hive_forwarder → rbee-hive
                                                ↓
                                         worker-catalog
                                                ↓
                                         Download/build binary
```

### Worker Process Management (Hive-Local)

```
rbee-keeper → queen-rbee → hive_forwarder → rbee-hive
                                                ↓
                                         spawn_worker()
                                                ↓
                                         Process spawned
                                                ↓
                                         Worker sends heartbeat to QUEEN
```

### Active Worker Tracking (Queen-Managed)

```
rbee-keeper → queen-rbee → Query worker registry
                                ↓
                         Workers tracked via heartbeats
                                ↓
                         Return active worker list
```

### Inference Routing (Queen-Managed)

```
rbee-keeper → queen-rbee → Query active workers
                                ↓
                         Select worker (scheduling)
                                ↓
                         Direct HTTP to worker
                                ↓
                         Stream response back
```

---

## 🔑 Key Distinctions

### Worker Process vs Active Worker

**Worker Process (Hive-Local):**
- Managed by hive's local process list (`ps`)
- Operations: Spawn, List (local), Get (local), Delete (kill)
- Hive doesn't track state - just executes commands
- Example: "List all worker processes on hive-gpu-01"

**Active Worker (Queen-Tracked):**
- Tracked by queen's heartbeat registry
- Operations: List (registry), Get (registry), Retire
- Queen maintains state from worker heartbeats
- Example: "List all active workers across all hives"

### Why This Separation?

1. **Hive is Stateless**
   - Hive doesn't maintain a registry
   - Hive just executes local commands
   - Simpler, more reliable

2. **Queen is Source of Truth**
   - Workers send heartbeats to queen
   - Queen knows which workers are alive
   - Queen can route inference requests

3. **Clear Responsibilities**
   - Hive: Local resource management
   - Queen: Distributed orchestration

---

## 📝 Implementation Status

### ✅ Implemented

- `should_forward_to_hive()` - Updated with all hive-local operations
- `Operation::name()` - All operation names defined
- `Operation::hive_id()` - Hive ID extraction for all operations

### ⚠️ TODO (Hive)

- Worker binary operations (download/build/list/get/delete)
- Worker process operations (list/get/delete)
- Model operations (download - provisioner needed)

### ⚠️ TODO (Queen)

- Active worker operations (list/get/retire)
- Worker registry (heartbeat tracking)
- Inference scheduling

---

## 🎯 Summary

**Total Operations:** 28
- **Hive-Local:** 13 operations
- **Queen-Managed:** 15 operations

**Architecture:**
- ✅ Clear separation of concerns
- ✅ Hive is stateless (local execution)
- ✅ Queen is stateful (distributed orchestration)
- ✅ Workers send heartbeats to queen (not hive)

**Next Steps:**
1. Implement worker binary operations in hive
2. Implement worker process operations in hive
3. Implement active worker registry in queen
4. Implement inference scheduling in queen

---

**TEAM-272 operation architecture complete! 🎉**

**28 operations defined, clearly categorized, ready for implementation!**
