# Declarative Lifecycle Migration Plan

**Date:** Oct 23, 2025  
**Status:** DESIGN / MIGRATION PLAN  
**Author:** TEAM-276

## Executive Summary

This document outlines the migration from **imperative lifecycle operations** to **declarative package manager pattern** for rbee v0.1.0+.

**Current:** Imperative operations (HiveInstall, HiveStart, WorkerSpawn, etc.)  
**Target:** Declarative config + sync operations (PackageSync, PackageStatus, etc.)

**Key Decision:** Queen manages both hive AND worker installation remotely. Hive only manages worker processes (start/stop), not installation.

---

## Architecture Shift

### Current Architecture (Imperative)

```
rbee-keeper → Operation::HiveInstall → queen-rbee → SSH → hive-host
                                                              ↓
                                                         Install hive binary

rbee-keeper → Operation::WorkerDownload → queen-rbee → HTTP → rbee-hive
                                                                  ↓
                                                            Download worker binary

rbee-keeper → Operation::WorkerSpawn → queen-rbee → HTTP → rbee-hive
                                                               ↓
                                                          Spawn worker process
```

**Problems:**
- ❌ Sequential operations
- ❌ No desired state
- ❌ Hive manages worker installation (complex)
- ❌ Manual coordination

### Target Architecture (Declarative)

```
~/.config/rbee/hives.conf → rbee sync → queen-rbee
                                           ↓
                                    Concurrent jobs:
                                    ┌─────────────┬─────────────┬─────────────┐
                                    ↓             ↓             ↓
                              Install hive-1  Install hive-2  Install hive-3
                              (via SSH)       (via SSH)       (via SSH)
                                    ↓             ↓             ↓
                              Install workers Install workers Install workers
                              (via SSH)       (via SSH)       (via SSH)
                                    ↓             ↓             ↓
                              Poll health     Poll health     Poll health
                                    ↓             ↓             ↓
                              Start hive      Start hive      Start hive
                                    ↓             ↓             ↓
                              ✅ Ready        ✅ Ready        ✅ Ready
```

**Benefits:**
- ✅ Concurrent installation (3-10x faster)
- ✅ Desired state management
- ✅ Queen manages everything remotely
- ✅ Hive is simpler (no worker installation)

---

## Key Architectural Decision

### Queen Manages Worker Installation Remotely

**Rationale:**
1. **Simpler hive** - Hive doesn't need worker installation logic
2. **Concurrent** - Queen can install workers on all hives in parallel
3. **Declarative** - Queen reads config, installs what's declared
4. **SSH already available** - Queen already SSHs to hive hosts

**New Responsibility Split:**

| Component | Responsibility |
|-----------|---------------|
| **Queen** | Install hive binary (SSH), Install worker binaries (SSH), Start/stop hive (SSH), Read declarative config |
| **Hive** | Spawn/kill worker processes (local), Track worker heartbeats, Report worker status, Manage worker catalog (read-only) |

**Worker Catalog:**
- Hive reads `~/.local/share/rbee/workers/` to see available workers
- Queen writes to this directory via SSH
- Hive never downloads workers itself

---

## Current Lifecycle Crates Analysis

### 1. queen-lifecycle (rbee-keeper → queen)

**Current Operations:**
- `install()` - Install queen binary
- `uninstall()` - Remove queen binary
- `start()` - Start queen daemon
- `stop()` - Stop queen daemon
- `ensure()` - Ensure queen is running
- `status()` - Check queen status
- `rebuild()` - Rebuild queen from source

**Migration:**
- ✅ **Keep:** `ensure()`, `start()`, `stop()`, `status()` - Still needed for queen daemon
- ❌ **Remove:** `install()`, `uninstall()` - Replaced by package manager
- ❌ **Remove:** `rebuild()` - Not needed (use pre-built binaries)

**New Operations:**
- None - Queen lifecycle stays imperative (it's local)

### 2. hive-lifecycle (queen → hive)

**Current Operations:**
- `install()` - Install hive binary via SSH
- `uninstall()` - Remove hive binary via SSH
- `start()` - Start hive daemon via SSH
- `stop()` - Stop hive daemon via SSH
- `ensure()` - Ensure hive is running
- `status()` - Check hive status
- `get()` - Get hive info
- `list()` - List all hives
- `capabilities()` - Get hive capabilities

**Migration:**
- ❌ **Remove:** `install()`, `uninstall()` - Replaced by declarative sync
- ✅ **Keep:** `start()`, `stop()`, `ensure()` - Still needed for daemon management
- ✅ **Keep:** `status()`, `get()`, `list()` - Still needed for queries
- ✅ **Keep:** `capabilities()` - Still needed for discovery

**New Operations:**
- `sync_hive()` - Sync single hive to desired state
- `sync_all_hives()` - Sync all hives concurrently

### 3. worker-lifecycle (hive → worker)

**Current Operations:**
- `install()` - Download worker binary
- `uninstall()` - Remove worker binary
- `start()` - Spawn worker process
- `stop()` - Kill worker process
- `get()` - Get worker info
- `list()` - List workers

**Migration:**
- ❌ **Remove:** `install()`, `uninstall()` - Queen installs workers via SSH
- ✅ **Keep:** `start()`, `stop()` - Hive still spawns/kills worker processes
- ✅ **Keep:** `get()`, `list()` - Hive still tracks worker processes

**New Operations:**
- None - Hive only manages processes, not binaries

### 4. daemon-lifecycle (shared utilities)

**Current Operations:**
- `ensure_daemon_running()` - Ensure daemon is running
- `ensure_daemon_with_handle()` - Ensure with handle pattern
- `poll_until_healthy()` - Poll health endpoint
- `install_daemon()` - Generic install
- `uninstall_daemon()` - Generic uninstall
- `start_http_daemon()` - Start HTTP daemon
- `stop_http_daemon()` - Stop HTTP daemon

**Migration:**
- ✅ **Keep all** - These are generic utilities
- ➕ **Add:** `sync_daemon()` - Sync daemon to desired state

---

## New Package Manager Operations

### Operations to Add

```rust
// bin/99_shared_crates/rbee-operations/src/lib.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Operation {
    // ... existing operations ...
    
    // ============================================================================
    // PACKAGE MANAGER OPERATIONS (Declarative)
    // ============================================================================
    
    /// Sync all hives and workers to match declarative config
    ///
    /// Reads ~/.config/rbee/hives.conf and ensures actual state matches desired state.
    /// Installs missing hives/workers, removes extra ones (if --remove-extra).
    ///
    /// **Handled by:** Queen (orchestrates concurrent sync jobs)
    /// **Concurrency:** All hives synced in parallel
    PackageSync {
        /// Path to config file (default: ~/.config/rbee/hives.conf)
        config_path: Option<String>,
        
        /// Dry run - show what would change without applying
        dry_run: bool,
        
        /// Remove components not in config
        remove_extra: bool,
        
        /// Force reinstall even if already installed
        force: bool,
    },
    
    /// Check if actual state matches declarative config
    ///
    /// Reports drift without making changes.
    PackageStatus {
        /// Path to config file
        config_path: Option<String>,
        
        /// Show detailed diff
        verbose: bool,
    },
    
    /// Install all components from declarative config
    ///
    /// Alias for PackageSync with specific flags.
    PackageInstall {
        /// Path to config file
        config_path: Option<String>,
        
        /// Force reinstall
        force: bool,
        
        /// Install only specific hive
        hive_alias: Option<String>,
    },
    
    /// Uninstall all components (or specific hive)
    PackageUninstall {
        /// Uninstall specific hive (None = all)
        hive_alias: Option<String>,
        
        /// Remove config files too
        purge: bool,
    },
    
    /// Validate declarative config without applying
    PackageValidate {
        /// Path to config file
        config_path: Option<String>,
    },
    
    /// Generate config from current state
    PackageMigrate {
        /// Output path for generated config
        output_path: String,
    },
}
```

### Operations to Remove

```rust
// REMOVE these operations (replaced by declarative sync):

Operation::HiveInstall { .. }      // → PackageSync
Operation::HiveUninstall { .. }    // → PackageSync / PackageUninstall
Operation::WorkerDownload { .. }   // → PackageSync
Operation::WorkerBuild { .. }      // → PackageSync
Operation::WorkerBinaryDelete { .. } // → PackageSync
```

### Operations to Keep

```rust
// KEEP these operations (still needed for runtime management):

// Hive daemon management
Operation::HiveStart { .. }
Operation::HiveStop { .. }
Operation::HiveList { .. }
Operation::HiveGet { .. }
Operation::HiveStatus { .. }
Operation::HiveCapabilities { .. }

// Worker process management (NOT installation)
Operation::WorkerSpawn { .. }
Operation::WorkerProcessList { .. }
Operation::WorkerProcessGet { .. }
Operation::WorkerProcessDelete { .. }

// Model management (on-demand)
Operation::ModelDownload { .. }
Operation::ModelList { .. }
Operation::ModelGet { .. }
Operation::ModelDelete { .. }

// Inference
Operation::Infer { .. }
```

---

## Declarative Config Schema

### ~/.config/rbee/rbee.conf

```toml
# Queen configuration
[queen]
# Queen mode: "standalone" or "attached-hive"
mode = "standalone"

# Queen port
port = 8500

# Auto-start on system boot
auto_start = true

# Package manager settings
[package]
# Concurrent installation limit (default: 10)
max_concurrent_installs = 10

# SSH timeout (seconds)
ssh_timeout = 30

# Health check timeout (seconds)
health_check_timeout = 60
```

### ~/.config/rbee/hives.conf

```toml
# Hive 1: Remote GPU server
[[hive]]
alias = "gpu-server-1"
hostname = "192.168.1.100"
ssh_user = "vince"
ssh_port = 22
hive_port = 8600

# Workers to install on this hive
# Queen will download these binaries via SSH
workers = [
    { type = "vllm", version = "latest" },
    { type = "llama-cpp", version = "latest" },
]

# Auto-start hive on sync
auto_start = true

# Hive 2: Another GPU server
[[hive]]
alias = "gpu-server-2"
hostname = "192.168.1.101"
ssh_user = "vince"
ssh_port = 22
hive_port = 8600

workers = [
    { type = "vllm", version = "latest" },
    { type = "comfyui", version = "latest" },
]

auto_start = true

# Hive 3: Local hive (for attached mode)
[[hive]]
alias = "local-hive"
hostname = "localhost"
ssh_user = "vince"
ssh_port = 22
hive_port = 8600

workers = [
    { type = "llama-cpp", version = "latest" },
]

auto_start = true
```

---

## Migration Path

### Phase 1: Add Declarative Config Support (8-12 hours)

**Goal:** Add config parsing without breaking existing code

**Tasks:**
1. Create `rbee-config/src/declarative.rs`
   - Parse `hives.conf`
   - Validate config
   - Provide typed structs

2. Add config validation
   - Check SSH connectivity
   - Check hostname resolution
   - Check worker types exist

**Files:**
- NEW: `bin/99_shared_crates/rbee-config/src/declarative.rs`
- MODIFIED: `bin/99_shared_crates/rbee-config/src/lib.rs`

**No breaking changes yet** - Just add config support

### Phase 2: Add Package Manager Operations (12-16 hours)

**Goal:** Add new operations to rbee-operations

**Tasks:**
1. Add `PackageSync`, `PackageStatus`, `PackageInstall`, etc. to Operation enum
2. Update `should_forward_to_hive()` - Package ops stay in queen
3. Add operation serialization/deserialization

**Files:**
- MODIFIED: `bin/99_shared_crates/rbee-operations/src/lib.rs`

**Breaking change:** New operations added, but old operations still work

### Phase 3: Implement Package Sync in Queen (24-32 hours)

**Goal:** Implement declarative sync logic in queen-rbee

**Tasks:**
1. Create `queen-rbee/src/package_manager/` module
   - `sync.rs` - Main sync logic
   - `diff.rs` - Compare actual vs desired state
   - `install.rs` - Install hives and workers concurrently
   - `status.rs` - Check current state

2. Implement concurrent installation
   - Use `tokio::spawn` for parallel SSH operations
   - Progress tracking
   - Error handling with partial success

3. Implement worker installation via SSH
   - Queen SSHs to hive host
   - Downloads worker binary to `~/.local/share/rbee/workers/`
   - Hive discovers workers from this directory

**Files:**
- NEW: `bin/10_queen_rbee/src/package_manager/` (entire module)
- MODIFIED: `bin/10_queen_rbee/src/job_router.rs` (add package operations)

**Key Implementation:**
```rust
// queen-rbee/src/package_manager/sync.rs

pub async fn sync_all_hives(config: HivesConfig, opts: SyncOptions) -> Result<SyncReport> {
    // 1. Query actual state (concurrent)
    let actual_state = query_actual_state(&config).await?;
    
    // 2. Compare with desired state
    let diff = compute_diff(&config, &actual_state);
    
    if opts.dry_run {
        return Ok(SyncReport::from_diff(diff));
    }
    
    // 3. Apply changes concurrently
    let sync_tasks: Vec<_> = diff.hives_to_install.iter().map(|hive| {
        tokio::spawn(sync_single_hive(hive.clone(), opts.clone()))
    }).collect();
    
    let results = futures::future::join_all(sync_tasks).await;
    
    Ok(SyncReport::from_results(results))
}

async fn sync_single_hive(hive: HiveConfig, opts: SyncOptions) -> Result<HiveSyncResult> {
    // 1. Install hive binary (if needed)
    if !is_hive_installed(&hive).await? {
        install_hive_binary(&hive).await?;
    }
    
    // 2. Install worker binaries (concurrent)
    let worker_tasks: Vec<_> = hive.workers.iter().map(|worker| {
        tokio::spawn(install_worker_binary(hive.clone(), worker.clone()))
    }).collect();
    
    futures::future::try_join_all(worker_tasks).await?;
    
    // 3. Start hive (if auto_start)
    if hive.auto_start {
        start_hive(&hive).await?;
    }
    
    // 4. Poll until healthy
    poll_until_healthy(&hive).await?;
    
    Ok(HiveSyncResult::success(hive.alias))
}

async fn install_worker_binary(hive: HiveConfig, worker: WorkerConfig) -> Result<()> {
    // Queen installs worker via SSH
    let ssh = SshClient::connect(&hive).await?;
    
    // Download worker binary to hive host
    let worker_url = format!("https://github.com/you/rbee/releases/download/{}/rbee-worker-{}", 
                             worker.version, worker.type_);
    
    ssh.execute(&format!(
        "mkdir -p ~/.local/share/rbee/workers && \
         curl -L {} -o ~/.local/share/rbee/workers/rbee-worker-{} && \
         chmod +x ~/.local/share/rbee/workers/rbee-worker-{}",
        worker_url, worker.type_, worker.type_
    )).await?;
    
    Ok(())
}
```

### Phase 4: Simplify Hive (Remove Worker Installation) (8-12 hours)

**Goal:** Remove worker installation logic from rbee-hive

**Tasks:**
1. Remove `worker-lifecycle/src/install.rs`
2. Remove `worker-lifecycle/src/uninstall.rs`
3. Update `worker-lifecycle` to only manage processes
4. Update hive to read worker catalog (read-only)

**Files:**
- DELETED: `bin/25_rbee_hive_crates/worker-lifecycle/src/install.rs`
- DELETED: `bin/25_rbee_hive_crates/worker-lifecycle/src/uninstall.rs`
- MODIFIED: `bin/25_rbee_hive_crates/worker-lifecycle/src/lib.rs`
- MODIFIED: `bin/20_rbee_hive/src/worker_catalog.rs` (read-only)

**Breaking change:** Hive can no longer install workers

### Phase 5: Update rbee-keeper CLI (8-12 hours)

**Goal:** Add package manager commands to rbee CLI

**Tasks:**
1. Add `rbee sync` command
2. Add `rbee status` command (package status)
3. Add `rbee install` command (alias for sync)
4. Add `rbee uninstall` command
5. Add `rbee validate` command
6. Deprecate old commands with warnings

**Files:**
- NEW: `bin/00_rbee_keeper/src/commands/sync.rs`
- NEW: `bin/00_rbee_keeper/src/commands/package_status.rs`
- MODIFIED: `bin/00_rbee_keeper/src/main.rs`

**Commands:**
```bash
# New commands
rbee sync                    # Sync all hives
rbee sync --dry-run          # Show what would change
rbee sync --hive gpu-1       # Sync specific hive
rbee status                  # Show package status
rbee install                 # Install all (alias for sync)
rbee uninstall               # Uninstall all
rbee validate                # Validate config

# Deprecated (still work, but warn)
rbee install-hive            # → Use 'rbee sync' instead
rbee install-worker          # → Use 'rbee sync' instead
```

### Phase 6: Remove Old Operations (4-6 hours)

**Goal:** Remove deprecated operations

**Tasks:**
1. Remove `HiveInstall`, `HiveUninstall` from Operation enum
2. Remove `WorkerDownload`, `WorkerBuild`, `WorkerBinaryDelete`
3. Update job_router to reject removed operations
4. Update documentation

**Files:**
- MODIFIED: `bin/99_shared_crates/rbee-operations/src/lib.rs`
- MODIFIED: `bin/10_queen_rbee/src/job_router.rs`

**Breaking change:** Old operations no longer work

---

## Updated Operation Flow

### Current (Imperative)

```
User: rbee install-hive --alias gpu-1
  ↓
rbee-keeper: Operation::HiveInstall { alias: "gpu-1" }
  ↓
queen-rbee: job_router → hive_lifecycle::install()
  ↓
SSH to gpu-1: Install hive binary
  ↓
Done (sequential, 30s)

User: rbee install-worker --hive gpu-1 --type vllm
  ↓
rbee-keeper: Operation::WorkerDownload { hive: "gpu-1", type: "vllm" }
  ↓
queen-rbee: Forward to hive
  ↓
rbee-hive: worker_lifecycle::install()
  ↓
Download worker binary
  ↓
Done (sequential, 20s)
```

**Total: 50 seconds for 1 hive + 1 worker**

### Target (Declarative)

```
User: rbee sync
  ↓
rbee-keeper: Operation::PackageSync { config_path: "~/.config/rbee/hives.conf" }
  ↓
queen-rbee: package_manager::sync_all_hives()
  ↓
Read hives.conf → 3 hives, 6 workers total
  ↓
Concurrent sync (3 hives in parallel):
  ┌─────────────────┬─────────────────┬─────────────────┐
  ↓                 ↓                 ↓
gpu-1:            gpu-2:            gpu-3:
- Install hive    - Install hive    - Install hive
- Install vllm    - Install vllm    - Install comfyui
- Install llama   - Install llama   - Start hive
- Start hive      - Start hive      - Poll health
- Poll health     - Poll health     - ✅ Ready (30s)
- ✅ Ready (30s)  - ✅ Ready (30s)
  ↓
Done (concurrent, 30s total)
```

**Total: 30 seconds for 3 hives + 6 workers (3x faster!)**

---

## Benefits of Declarative Approach

### 1. **Concurrent Installation** ⭐⭐⭐
- 3 hives: 30s instead of 90s (3x faster)
- 10 hives: 30s instead of 300s (10x faster)

### 2. **Simpler Hive** ⭐⭐⭐
- Hive doesn't install workers
- Hive only manages processes
- Less code, fewer bugs

### 3. **Centralized Control** ⭐⭐⭐
- Queen manages all installation
- Single source of truth (config file)
- Easier to reason about

### 4. **Desired State** ⭐⭐⭐
- Config declares what should exist
- `rbee sync` makes it so
- Drift detection automatic

### 5. **Better for Scheduler** ⭐⭐
- Queen knows which workers are available
- No need to query hive for worker list
- Faster scheduling decisions

---

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1: Config Support | 8-12 hours | Config parsing works |
| Phase 2: New Operations | 12-16 hours | PackageSync operation added |
| Phase 3: Package Manager | 24-32 hours | `rbee sync` works |
| Phase 4: Simplify Hive | 8-12 hours | Hive no longer installs workers |
| Phase 5: Update CLI | 8-12 hours | New commands available |
| Phase 6: Remove Old Ops | 4-6 hours | Old operations removed |
| **Total** | **64-90 hours** | **Declarative lifecycle complete** |

**Estimated:** 2-3 weeks of focused work

---

## Risks & Mitigation

### Risk 1: Breaking Changes

**Risk:** Existing users must migrate

**Mitigation:**
- Provide `rbee migrate` command to generate config
- Support old operations with deprecation warnings (Phase 5)
- Remove old operations only in Phase 6 (after migration period)

### Risk 2: SSH Complexity

**Risk:** SSH operations can fail (network, auth, etc.)

**Mitigation:**
- Robust error handling with retries
- Partial success reporting (some hives succeed, some fail)
- `rbee sync --retry-failed` to retry only failed hives

### Risk 3: Concurrent Installation Bugs

**Risk:** Parallel SSH operations may have race conditions

**Mitigation:**
- Thorough testing with multiple hives
- Progress tracking to debug failures
- Dry-run mode to test without applying

### Risk 4: Worker Catalog Sync

**Risk:** Hive's worker catalog may be stale

**Mitigation:**
- Hive scans `~/.local/share/rbee/workers/` on startup
- Hive re-scans on demand (new operation: `WorkerCatalogRefresh`)
- Queen can trigger refresh after installing workers

---

## Conclusion

**Declarative lifecycle is the right architecture:**

✅ **Simpler** - Hive doesn't install workers  
✅ **Faster** - Concurrent installation  
✅ **Better** - Desired state management  
✅ **Scalable** - Works with 100+ hives  

**Migration is feasible:**
- 6 phases, 64-90 hours
- Gradual migration (support both during transition)
- Clear benefits justify the effort

**Next Steps:**
1. Review this plan
2. Start Phase 1 (config support)
3. Implement incrementally
4. Test thoroughly at each phase

---

**Created by:** TEAM-276  
**Status:** Design document  
**Next:** Implement Phase 1 (config support)
